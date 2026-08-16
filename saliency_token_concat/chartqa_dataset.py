import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
TEXT_TOKEN_TYPE = 0


SYSTEM_PROMPT_TEXT = (
    "You are an expert chart question answering assistant.\n"
    "You will be given a chart image.\n"
    "Answer the question only based on the given images. Do not use external knowledge or assumptions.\n"
    "Return ONLY the final answer. Do not include explanation or reasoning.\n"
)

USER_TEMPLATE_TEXT = "Answer this question based on the image: {question}\n\n"


class ChartQASaliencyDataset(Dataset):
    def __init__(self, json_path, img_dir, processor, max_samples=None):
        with open(json_path) as f:
            raw = json.load(f)

        self.img_dir = Path(img_dir)
        self.processor = processor  # Qwen3-VL AutoProcessor, same instance as in train.py

        self.items = []
        n_missing = 0
        for r in raw:
            img_path = self.img_dir / r["imgname"]
            if not img_path.exists():
                n_missing += 1
                continue
            self.items.append(r)
        if n_missing:
            print(f"WARNING: {n_missing} records skipped, chart image not found in {img_dir}")

        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        raw_image = Image.open(self.img_dir / item["imgname"]).convert("RGB")
        question = item["query"]
        answer = item["label"]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": USER_TEMPLATE_TEXT.format(question=question)},
                ],
            },
        ]

        # Single call, matching Qwen3VL.generate() exactly (tokenize=True,
        # return_dict=True) instead of a two-step apply_chat_template +
        # processor(...) call -- this is the form you've already confirmed
        # works end-to-end with this processor version.
        prompt_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"][0].tolist()

        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos_id = self.processor.tokenizer.eos_token_id

        input_ids = prompt_ids + answer_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]  # only the answer contributes to loss

        mm_token_type_ids = prompt_inputs["mm_token_type_ids"][0].tolist()
        mm_token_type_ids = mm_token_type_ids + [TEXT_TOKEN_TYPE] * (len(answer_ids) + 1) #asnwer id + [eos]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": prompt_inputs["pixel_values"],
            "image_grid_thw": prompt_inputs["image_grid_thw"],
            "raw_image": raw_image,
            "question_text": question,
        }