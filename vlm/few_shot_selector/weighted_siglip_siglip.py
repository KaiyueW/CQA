import os
os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"

import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging

from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)


class ChartQATrainDataset(Dataset):
    def __init__(self, chart_dir, json_path):
        self.chart_dir = chart_dir
        with open(json_path, "r") as f:
            self.samples = json.load(f)

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
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


class ChartQAKNNRetriever:
    """
    Single-stage weighted-fusion few-shot example retriever for ChartQA.

    Score = w1 * cosine(image_sim) + w2 * cosine(text_sim)
    """

    def __init__(
        self,
        train_dataset,
        device,
        batch_size=32,
        vision_encoder_path="google/siglip-base-patch16-224",
        cached_image_features=None,
        cached_text_features=None,          
        w1=0.5,                             
        w2=0.5,
        text_batch_size=100,
    ):
        self.dataset = train_dataset
        self.device = device
        self.batch_size = batch_size
        self.text_batch_size = text_batch_size
        self.w1 = w1                       
        self.w2 = w2                       

        self.processor = AutoProcessor.from_pretrained(vision_encoder_path)
        self.model = AutoModel.from_pretrained(vision_encoder_path).to(self.device).eval()
        print(f"-----Loaded SigLIP model ({vision_encoder_path}) for image + text encoding.")

        # ---- precompute image features for the whole training pool ----
        if cached_image_features is None:
            self.image_features = self._precompute_image_features()
        else:
            self.image_features = cached_image_features

        # precompute text features for the whole training pool
        if cached_text_features is None:
            self.text_features = self._precompute_text_features()
        else:
            self.text_features = cached_text_features

    def _encode_images_siglip(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)  # [Batch_size, Dimension]
        feats = out.pooler_output 
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu()

    def _encode_texts_siglip(self, texts, batch_size=None):
        batch_size = batch_size or self.text_batch_size
        all_feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                feats = out.pooler_output
            all_feats.append(feats.detach().cpu())
        feats = torch.cat(all_feats, dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _precompute_image_features(self):
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
                feats = self._encode_images_siglip(batch["image"])
                features.append(feats)
        return torch.cat(features)

    def _precompute_text_features(self):
        print("-----Precomputing training text features.")
        all_queries = [self.dataset[i]["query"] for i in range(len(self.dataset))]
        return self._encode_texts_siglip(all_queries)

    def save_image_features_to_path(self, path):
        torch.save(self.image_features, path)

    def save_text_features_to_path(self, path):
        torch.save(self.text_features, path)

    def retrieve(self, test_image, test_query, num_examples, do_reverse=False):
        """
        Retrieve few-shot examples for a given test image and query,
        using fused Score = w1 * cosine(image) + w2 * cosine(text)
        """
        with torch.no_grad():
            # ---- image similarity over the whole training pool ----
            query_img_feat = self._encode_images_siglip([test_image])          # [1, D_img]
            img_sim = (query_img_feat @ self.image_features.T).squeeze(0)      # [N_train]

            # ---- text similarity over the whole training pool ----
            query_text_feat = self._encode_texts_siglip([test_query])          # [1, D_text]
            text_sim = (query_text_feat @ self.text_features.T).squeeze(0)     # [N_train]

            # Both similarity vectors are already cosine similarities in [-1, 1] since both feature sets are L2-normalized,
            fused_score = self.w1 * img_sim + self.w2 * text_sim              # [N_train]

            num_examples = min(num_examples, fused_score.shape[0])
            final_indices = fused_score.argsort(descending=True)[:num_examples]
            final_indices = final_indices.tolist()

            if do_reverse:
                final_indices = list(reversed(final_indices))

            return [self.dataset[i] for i in final_indices] # return "img", "imgname", "query", "label", "saliency_map" for each retrieved example


def run(
    train_chart_dir,
    train_json,
    test_chart_dir,
    test_json,
    output_json,
    num_shots=4,
    batch_size=32,
    cached_image_features=None,
    save_image_features=None,
    cached_text_features=None,      
    save_text_features=None,       
    w1=0.5,                      
    w2=0.5,                       
    do_reverse=False,
    device=None,
):
    """Build the retriever once, then run it over every sample in test_json and write results to output_json."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChartQATrainDataset(chart_dir=train_chart_dir, json_path=train_json)

    cached_img_feats = None
    if cached_image_features is not None and os.path.exists(cached_image_features):
        print(f"Loading cached training image features from {cached_image_features}")
        cached_img_feats = torch.load(cached_image_features)

    cached_txt_feats = None
    if cached_text_features is not None and os.path.exists(cached_text_features):
        print(f"Loading cached training text features from {cached_text_features}")
        cached_txt_feats = torch.load(cached_text_features)

    retriever = ChartQAKNNRetriever(
        train_dataset=train_dataset,
        device=device,
        batch_size=batch_size,
        cached_image_features=cached_img_feats,
        cached_text_features=cached_txt_feats, 
        w1=w1,                                    
        w2=w2,                                  
    )

    if save_image_features is not None:
        feat_dir = os.path.dirname(os.path.abspath(save_image_features))
        os.makedirs(feat_dir, exist_ok=True)
        retriever.save_image_features_to_path(save_image_features)

    if save_text_features is not None:
        feat_dir = os.path.dirname(os.path.abspath(save_text_features))
        os.makedirs(feat_dir, exist_ok=True)
        retriever.save_text_features_to_path(save_text_features)

    with open(test_json, "r") as f:
        test_samples = json.load(f)

    results = {}
    for sample in tqdm(test_samples, desc="Retrieving few-shot examples for test set"):
        test_image = Image.open(os.path.join(test_chart_dir, sample["imgname"])).convert("RGB")

        examples = retriever.retrieve(
            test_image=test_image,
            test_query=sample["query"],
            num_examples=num_shots,
            do_reverse=do_reverse,
        )

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
        output_json="output/weighted_siglip_siglip_fewshot_examples.json",
        num_shots=8,
        cached_image_features="siglip_train_image_feats.pt",
        save_image_features="siglip_train_image_feats.pt",
        cached_text_features="siglip_train_text_feats.pt",
        save_text_features="siglip_train_text_feats.pt", 
        w1=0.5, 
        w2=0.5,  
    )