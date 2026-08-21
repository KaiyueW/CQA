import argparse
import json
import os
from pathlib import Path

os.environ["HF_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

from qwen3vl_saliency_model import Qwen3VLWithSaliencyBottleneck
from data_collator import add_saliency_token, NUM_SALIENCY_TOKENS

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_END_TOKEN_STR = "<|vision_end|>"

SYSTEM_PROMPT_TEXT = (
    "You are an expert chart question answering assistant.\n"
    "You will be given a chart image.\n"
    "Answer the question only based on the given images. Do not use external knowledge or assumptions.\n"
    "Return ONLY the final answer. Do not include explanation or reasoning.\n"
)
USER_TEMPLATE_TEXT = "Answer this question based on the image: {question}\n\n"


def load_model_and_tokenizer(checkpoint_dir: str, attn_implementation: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vision_end_token_id = tokenizer.convert_tokens_to_ids(VISION_END_TOKEN_STR)

    base_model = Qwen3VLWithSaliencyBottleneck.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn_implementation,
    )
    saliency_token_id = add_saliency_token(tokenizer, model=base_model)

    base_model.attach_saliency_modules(
        saliency_token_id=saliency_token_id,
        llm_hidden_size=base_model.config.text_config.hidden_size,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    model.to("cuda")

    return model, tokenizer, processor, saliency_token_id, vision_end_token_id


def build_and_splice_inputs(processor, raw_image, question, vision_end_token_id, saliency_token_id):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": USER_TEMPLATE_TEXT.format(question=question)},
            ],
        },
    ]
    prompt_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )

    ids = prompt_inputs["input_ids"][0].tolist()
    mm_token_type_ids = prompt_inputs["mm_token_type_ids"][0].tolist()

    idx = ids.index(vision_end_token_id) + 1
    ids = ids[:idx] + [saliency_token_id] * NUM_SALIENCY_TOKENS + ids[idx:]
    mm_token_type_ids = mm_token_type_ids[:idx] + [0] * NUM_SALIENCY_TOKENS + mm_token_type_ids[idx:]

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([mm_token_type_ids], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": prompt_inputs["pixel_values"],
        "image_grid_thw": prompt_inputs["image_grid_thw"],
    }


@torch.no_grad()
def generate_answer(model, tokenizer, inputs, saliency_latent, max_new_tokens=20):
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    saliency_latents = saliency_latent.unsqueeze(0).to("cuda")  # [1, 49, 768]

    output_ids = model.generate(
        **inputs,
        saliency_latents=saliency_latents,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="trained LoRA+projector checkpoint dir")
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--test_img_dir", required=True)
    parser.add_argument("--test_latent_dir", required=True,
                         help="output dir from precompute_saliency_latents.py run on the TEST split")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    model, tokenizer, processor, saliency_token_id, vision_end_token_id = \
        load_model_and_tokenizer(args.checkpoint_dir)

    with open(args.test_json) as f:
        samples = json.load(f)[:args.max_samples]

    img_dir = Path(args.test_img_dir)
    latent_dir = Path(args.test_latent_dir)

    results = []
    n_skipped = 0
    for i, sample in enumerate(samples):
        img_path = img_dir / sample["imgname"]
        latent_path = latent_dir / f"{Path(sample['saliency_map']).stem}.pt"
        if not img_path.exists() or not latent_path.exists():
            n_skipped += 1
            continue

        raw_image = Image.open(img_path).convert("RGB")
        question = sample["query"]

        inputs = build_and_splice_inputs(
            processor, raw_image, question, vision_end_token_id, saliency_token_id,
        )
        saliency_latent = torch.load(latent_path)  # [49, 768]

        pred_answer = generate_answer(model, tokenizer, inputs, saliency_latent, args.max_new_tokens)

        results.append({
            "imgname": sample["imgname"],
            "saliency_map": sample["saliency_map"],
            "question": question,
            "gt_answer": sample["label"],
            "pred_answer": pred_answer,
            "is_numerical": sample["is_numerical"],
            "is_year": sample["is_year"],
        })

        if (i + 1) % 50 == 0:
            print(f"----------{i + 1}/{len(samples)} processed.----------")

    if n_skipped:
        print(f"WARNING: skipped {n_skipped} samples (missing image or precomputed latent)")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions -> {args.out_path}")


if __name__ == "__main__":
    main()

# python inference.py --checkpoint_dir ./finetune/checkpoints/saliency_bottleneck/checkpoint-2800 --test_json ../data/ChartQA_data/test/test_all_preprocessed.json --test_img_dir ../data/ChartQA_data/test/png --test_latent_dir ./saliency_latents/test --out_path ./results/test.json  --max_samples 5
#
# python evaluation.py --result_path ./results/qwen3vl_saliency_tokenconcat_zeroshot_all3200.json