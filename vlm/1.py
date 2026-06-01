import torch
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

# store paths
os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"]= "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model

# Paths 
TRAIN_JSON      = "../ChartQA_data/train/train_human.json"
TEST_JSON       = "../ChartQA_data/test/test_human.json"
TRAIN_IMG_DIR   = "../ChartQA_data/train/png"
TEST_IMG_DIR    = "../ChartQA_data/test/png"
TRAIN_HEATMAP   = "../saliency_maps/ChartQA_train"
TEST_HEATMAP    = "../saliency_maps/ChartQA_test"
MAX_SAMPLES     = 50


# Prompt builders 
def build_prompt_zeroshot(question: str, use_saliency: bool) -> str:
    if use_saliency:
        return (
            "USER: <image>\n<image>\n"
            "The first image is a chart. "
            "The second image is a saliency map highlighting regions relevant to the question.\n"
            "Use the saliency map to guide your attention to answer the following question.\n\n"
            f"Question: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )
    else:
        return (
            "USER: <image>\n"
            f"Answer this question based on the image: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )


def build_prompt_fewshot(question: str, examples: list, use_saliency: bool) -> tuple[str, list]:
    images = []

    if use_saliency:
        prompt = (
            "USER: You need to answer a question based on a chart and its saliency map. "
            "The saliency map highlights the regions of the chart most relevant to the question. "
            "Here are some examples for your reference:\n\n"
        )
    else:
        prompt = (
            "USER: You need to answer a question based on a chart. "
            "Here are some examples for your reference:\n\n"
        )

    for idx, ex in enumerate(examples, start=1):
        if use_saliency:
            prompt += (
                f"{idx}. Given this chart <image> and this saliency map <image>, "
                f"the question is: \"{ex['query']}\", "
                f"the answer is: {ex['label']}.\n\n"
            )
            images.append(ex["chart_img"])
            images.append(ex["heatmap_img"])
        else:
            prompt += (
                f"{idx}. Given this chart <image>, "
                f"the question is: \"{ex['query']}\", "
                f"the answer is: {ex['label']}.\n\n"
            )
            images.append(ex["chart_img"])

    if use_saliency:
        prompt += (
            "Now, similar to the examples above, answer the following question.\n"
            "This is the chart <image> and this is its saliency map <image>.\n"
            f"Question: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )
    else:
        prompt += (
            "Now, similar to the examples above, answer the following question.\n"
            "This is the chart <image>.\n"
            f"Question: {question}\n"
            "Give a short, direct answer only.\n"
            "ASSISTANT:"
        )

    return prompt, images


# Few-shot example retrieval, here we simply retrieve other questions for the same chart.
def retrieve_examples(train_samples, current_sample) -> list:
    current_stem = os.path.splitext(current_sample["imgname"])[0] # "chart001.png" → "chart001"
    result = []

    for s in train_samples:
        same_image = os.path.splitext(s["imgname"])[0] == current_stem
        different_query = s["query"] != current_sample["query"]
        
        if same_image and different_query:
            result.append(s)
            
    return result # json item for the train examples with keys: "imgname", "query", "label"


def load_example_images(examples, img_dir, heatmap_dir):
    counter = defaultdict(int)
    loaded  = []
    for ex in examples:
        stem  = os.path.splitext(ex["imgname"])[0]
        q_idx = counter[stem]
        counter[stem] += 1
        chart_img   = Image.open(os.path.join(img_dir, ex["imgname"])).convert("RGB")
        heatmap_img = Image.open(os.path.join(heatmap_dir, f"{stem}_Q{q_idx}.png")).convert("RGB")
        loaded.append({
            **ex,
            "chart_img":   chart_img,
            "heatmap_img": heatmap_img,
        })
    return loaded #json item with keys: "imgname", "query", "label", "chart_img", "heatmap_img" (real images)



def run_inference(model, samples, train_samples, setting, use_saliency):
    results       = []
    query_counter = defaultdict(int)

    for i, sample in enumerate(samples):
        # load from test json file
        imgname   = sample["imgname"]
        question  = sample["query"]
        gt_answer = sample["label"]

        stem  = os.path.splitext(imgname)[0]
        q_idx = query_counter[stem]
        query_counter[stem] += 1

        chart_img   = Image.open(os.path.join(TEST_IMG_DIR, imgname)).convert("RGB")
        heatmap_img = Image.open(os.path.join(TEST_HEATMAP, f"{stem}_Q{q_idx}.png")).convert("RGB") if use_saliency else None

        if setting == "zeroshot":
            prompt = build_prompt_zeroshot(question, use_saliency)
            images = [chart_img, heatmap_img] if use_saliency else [chart_img]

        elif setting == "fewshot":
            raw_examples = retrieve_examples(train_samples, sample)
            examples     = load_example_images(raw_examples, TRAIN_IMG_DIR, TRAIN_HEATMAP)
            prompt, ex_images = build_prompt_fewshot(question, examples, use_saliency)
            images = ex_images + ([chart_img, heatmap_img] if use_saliency else [chart_img])

        print(f"\n{'='*60}")
        print(f"Sample {i+1} | {imgname} | setting: {setting}")
        print(f"{'='*60}")
        print(prompt)
        print(f"{'='*60}\n")
        predicted_answer = model.generate(prompt, images)

        results.append({
            "imgname":      imgname,
            "saliency_map":      f"{stem}_Q{q_idx}.png" if use_saliency else None,
            "question":     question,
            "gt_answer":    gt_answer,
            "pred_answer":  predicted_answer,
            })

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(samples)} processed.")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava15", "internvl"], default="llava15")
    parser.add_argument("--setting", choices=["zeroshot", "fewshot"],  default="zeroshot")
    parser.add_argument("--use_saliency", action="store_true")
    parser.add_argument("--max_samples",  type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    saliency_tag = "with_saliency" if args.use_saliency else "no_saliency"
    output_path  = f"./result_jsons/{args.model}_{args.setting}_{args.use_saliency}.json"

    with open(TEST_JSON, "r") as f:
        samples = json.load(f)[:args.max_samples] # load test samples, samples[0]["imgname"] = "1.png"

    train_samples = []
    if args.setting == "fewshot":
        with open(TRAIN_JSON, "r") as f:
            train_samples = json.load(f) # load train samples for few-shot retrieval

    model   = load_model(args.model)
    results = run_inference(model, samples, train_samples, args.setting, args.use_saliency)

    Path("./result_jsons").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# python 1.py --model llava15  --setting zeroshot --use_saliency
