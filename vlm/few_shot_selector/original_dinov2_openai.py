import os
os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"

import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging

from transformers import AutoImageProcessor, AutoModel
from openai import OpenAI

logger = logging.getLogger(__name__)


class ChartQATrainDataset(Dataset): #dataset loader
    def __init__(self, chart_dir, json_path):
        self.chart_dir = chart_dir
        with open(json_path, "r") as f:
            self.samples = json.load(f)[:10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chart_path = os.path.join(self.chart_dir, sample["imgname"])
        chart = Image.open(chart_path).convert("RGB")
        return {
            "image": chart,
            "imgname": sample["imgname"],
            "query": sample["query"],
            "label": sample["label"],
            "saliency_map": sample["saliency_map"],
        }


def custom_collate_fn(batch):
    """Collate a list of dicts into a dict of lists."""
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch
    # {"image": [img1, img2, ...], "query": [q1, q2, ...]}


class ChartQAKNNRetriever:
    """
    Two-stage kNN few-shot example retriever for ChartQA.

    Stage 1 (image): retrieve the top `image_topk` most image-similar training
                       examples using a DINOv2 image encoder (CLS token).
    Stage 2 (text):    from those candidates, re-rank by question-text similarity
                       (OpenAI embeddings) and keep the top `num_examples`.
    """

    def __init__(
        self,
        train_dataset,
        device,
        batch_size=32,
        vision_encoder_path="facebook/dinov2-base",
        text_model_name="text-embedding-3-small",
        cached_image_features=None,
        image_topk=200,
        openai_api_key=None,          # === CHANGED: new param, falls back to OPENAI_API_KEY env var ===
        text_batch_size=100, 
    ):
        self.dataset = train_dataset
        self.device = device
        self.batch_size = batch_size
        self.image_topk = image_topk
        self.text_model_name = text_model_name          # === CHANGED ===
        self.text_batch_size = text_batch_size            # === CHANGED ===

        self.image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
        self.image_model = AutoModel.from_pretrained(vision_encoder_path).to(self.device).eval()
        print(f"-----Loaded DINOv2 image encoder ({vision_encoder_path}).")

        self.openai_client = OpenAI(api_key=openai_api_key)  # uses OPENAI_API_KEY env var if not passed
        print(f"-----Using OpenAI embedding model ({text_model_name}).")

        # ---- precompute image features for the whole training pool ----
        if cached_image_features is None:
            self.image_features = self._precompute_image_features()
        else:
            self.image_features = cached_image_features


    def _encode_images_dinov2(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
        cls_feats = outputs.last_hidden_state[:, 0, :]  # get the CLS token at the very front [0], with shape [Batch_size, Dimension]
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
        return cls_feats.detach().cpu()


    def _encode_texts_openai(self, texts):
        all_feats = []
        for i in range(0, len(texts), self.text_batch_size):
            batch = texts[i:i + self.text_batch_size]
            response = self.openai_client.embeddings.create(
                model=self.text_model_name,
                input=batch,
            )
            batch_feats = torch.tensor([item.embedding for item in response.data])
            all_feats.append(batch_feats)
        feats = torch.cat(all_feats, dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats  # [Num_texts, Dimension], cpu tensor

    def _precompute_image_features(self):
        """Encode every training image and cache the normalized feature matrix [Num_train, D]."""
        features = []
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            num_workers=8,
        )
        print("-----Precomputing training image features.")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Precomputing training image features"):
                feats = self._encode_images_dinov2(batch["image"])
                features.append(feats)
        return torch.cat(features) #concat all batches into a single tensor of shape [Num_train, D]

    def save_image_features_to_path(self, path):
        """Persist precomputed features so you don't recompute them every run."""
        torch.save(self.image_features, path)

    def retrieve(self, test_image, test_query, num_examples, do_reverse=False):
        """
        Retrieve the top `num_examples` training examples for ONE test sample.

        Args:
            test_image (PIL.Image): the test chart image.
            test_query (str): the test question text.
            num_examples (int): how many few-shot examples to return.
            do_reverse (bool): if True, return with the closest match LAST
                                (handy when building a prompt where you want
                                the most relevant example nearest the query).

        Returns:
            List[dict]: each dict has keys "imgname", "query", "label",
                        taken directly from the training dataset.
        """
        with torch.no_grad():
            # ---- Stage 1: image similarity over the whole training pool ----
            query_img_feat = self._encode_images_dinov2([test_image])  # [1, D]

            img_sim = (query_img_feat @ self.image_features.T).squeeze(0)  # [N_train]

            top_indices = img_sim.argsort(descending=True)[:self.image_topk]
            candidates = [self.dataset[i] for i in top_indices.tolist()]

            # ---- Stage 2: re-rank the candidates by text similarity ----
            query_text_feat = self._encode_texts_openai([test_query])  # [1, D]

            cand_texts = [c["query"] for c in candidates]
            cand_text_feats = self._encode_texts_openai(cand_texts)  # [topk, D]

            text_sim = (query_text_feat @ cand_text_feats.T).squeeze(0)  # [topk]

            num_examples = min(num_examples, text_sim.shape[0]) # should be 200 candidates.
            final_indices = text_sim.argsort(descending=True)[:num_examples]
            final_indices = final_indices.tolist()

            if do_reverse:
                final_indices = list(reversed(final_indices))

            return [candidates[i] for i in final_indices]


# Batch entry point: retrieve few-shot examples for every sample in a test json
def run(
    train_chart_dir,
    train_json,
    test_chart_dir,
    test_json,
    output_json,
    num_shots=4,
    image_topk=200,
    batch_size=32,
    cached_image_features=None,
    save_image_features=None,
    do_reverse=False,
    device=None,
    openai_api_key=None,   # === CHANGED: new param, passed through to retriever ===
):
    """Build the retriever once, then run it over every sample in test_json and write results to output_json."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChartQATrainDataset(chart_dir=train_chart_dir, json_path=train_json)

    cached_features = None
    if cached_image_features is not None and os.path.exists(cached_image_features):
        print(f"Loading cached training image features from {cached_image_features}")
        cached_features = torch.load(cached_image_features)

    retriever = ChartQAKNNRetriever(
        train_dataset=train_dataset,
        device=device,
        batch_size=batch_size,
        cached_image_features=cached_features,
        image_topk=image_topk,
        openai_api_key=openai_api_key,   # === CHANGED ===
    )

    if save_image_features is not None:
        feat_dir = os.path.dirname(os.path.abspath(save_image_features))
        os.makedirs(feat_dir, exist_ok=True)
        retriever.save_image_features_to_path(save_image_features)
 

    with open(test_json, "r") as f:
        test_samples = json.load(f)[:5]

    results = {}
    for sample in tqdm(test_samples, desc="Retrieving few-shot examples for test set"):
        test_image = Image.open(os.path.join(test_chart_dir, sample["imgname"])).convert("RGB")

        examples = retriever.retrieve(
            test_image=test_image,
            test_query=sample["query"],
            num_examples=num_shots,
            do_reverse=do_reverse,
        )

        # keep only the identifying fields, not PIL images, so the json stays small
        results[sample['saliency_map']] = [
            {"imgname": ex["imgname"], "query": ex["query"], "label": ex["label"], "saliency_map": ex["saliency_map"]}
            for ex in examples
        ]

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote kNN few-shot examples for {len(results)} test samples to {output_json}")
    return results


if __name__ == "__main__":

    run(
        train_chart_dir="../data/ChartQA_data/train/png",
        train_json="../data/ChartQA_data/train/train_human_preprocessed.json",
        test_chart_dir="../data/ChartQA_data/test/png",
        test_json="../data/ChartQA_data/test/test_human_preprocessed.json",
        output_json="output/dinov2_openai_knn_fewshot_examples.json",
        num_shots=8,
        image_topk=200,
        cached_image_features="dinov2_train_image_feats.pt",
        save_image_features="dinov2_train_image_feats.pt",
    )