import torch
import json
import os
import argparse
from pathlib import Path
from PIL import Image
import random

# store paths
os.environ["HF_HOME"]       = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"]  = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"]= "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_model

# Paths 
TRAIN_JSON      = "../data/ChartQA_data/train/train_human_preprocessed.json" # Note: we use the same test json for training samples in few-shot setting, but we will retrieve different questions for the same chart as examples.
TEST_JSON       = "../data/ChartQA_data/test/test_human_preprocessed.json"
TRAIN_IMG_DIR   = "../data/ChartQA_data/train/png"
TEST_IMG_DIR    = "../data/ChartQA_data/test/png"
TRAIN_HEATMAP   = "../data/saliency_maps/ChartQA_train"
TEST_HEATMAP    = "../data/saliency_maps/ChartQA_test" # the saliency map dir for inference, you can change to the one you want.
MAX_SAMPLES     = 100



# Prompt builders 
def build_prompt_zeroshot(question: str, chart_img, heatmap_img=None) -> list:
    system = {
        "role": "system",
        "content": [
            {"type": "text", "text": 
            "You are an expert chart analysis assistant.\n"
            "Your task is to provide the precise final answer to the user's question.\n"
            "Only return the final answer in a concise format that directly answers the question.\n"}
            ]
    }

    if heatmap_img is not None:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "image", "image": heatmap_img},
                {"type": "text", "text": 
                    "The first image is a chart. "
                    "The second image is a saliency map highlighting regions relevant to the question.\n"
                    "Use the saliency map to guide your attention to answer the following question.\n"
                    f"Question: {question}\n\n"
                    "Answer with only the final answer, don't provide any explanation or reasoning steps."
                }
            ]
        }
    else:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "text", "text": 
                f"Answer this question based on the chart: {question}\n\n"
                "Your answer must contain ONLY the short final answer. Don't provide any explanation or reasoning steps."}
            ]
        }

    return [system, user]

def build_prompt_fewshot(question: str, examples: list, chart_img, heatmap_img=None) -> list:

    prompt = [{
        "role": "system",
        "content": [
            {"type": "text", "text": 
            "You are an expert chart analysis assistant.\n"
            "Your task is to provide the precise final answer to the user's question.\n"
            "Only return the final answer in a concise format that directly answers the question.\n"
            "You will be given several examples for your reference before answering the final question."}
            ]
    }]

    for ex in examples:
        user_content = []
        if heatmap_img is not None:
            user_content.append({"type": "image", "image": ex["chart_img"]})
            user_content.append({"type": "image", "image": ex["heatmap_img"]})
            user_content.append({"type": "text", "text":
                "Given the chart and its saliency map which highlights the regions of the chart most relevant to the question. "
                f"Answer the following question: {ex['query']}\n"})
        else:
            user_content.append({"type": "image", "image": ex["chart_img"]})
            user_content.append({"type": "text", "text":
                f"Given the chart, answer the following question: {ex['query']}\n"})
        
        prompt.append({"role": "user", 
                       "content": user_content})

        prompt.append({"role": "assistant",
                        "content": [{"type": "text", "text": ex['label']}]})

    user_content = []   
    if heatmap_img is not None:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "image", "image": heatmap_img})
        user_content.append({"type": "text", "text":
            "Now, similar to the examples above.\n"
            "Given this chart and its saliency map which highlights the regions of the chart most relevant to the question.\n"
            "Use the saliency map to guide your attention to answer the following question.\n"
            f"Question: {question}\n"
            "Your answer must contain ONLY the short final answer. Don't provide any explanation or reasoning steps."})
    else:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "text", "text":
            "Now, similar to the examples above, answer the following question based on the given chart.\n"
            f"Question: {question}\n"
            "Your answer must contain ONLY the short final answer. Don't provide any explanation or reasoning steps."})
    
    prompt.append({"role": "user", "content": user_content})

    return prompt


# Few-shot example retrieval, here we simply retrieve other questions for the same chart.
def retrieve_examples(train_samples, current_sample) -> list:
    result = []

    s = random.choice(train_samples)
    result.append(s)
    print(f"Retrieved example for {current_sample['imgname']}: {s['imgname']}, {s['query']}, {s['label']}")
    
    return result # json item for the train examples with keys: "imgname", "query", "label", "is_numerical", "saliency_map" 


def load_example_images(examples, img_dir, heatmap_dir):
    loaded  = []
    for ex in examples:
        chart_img   = Image.open(os.path.join(img_dir, ex["imgname"])).convert("RGB")
        heatmap_img = Image.open(os.path.join(heatmap_dir, ex["saliency_map"])).convert("RGB")
        loaded.append({
            **ex,
            "chart_img":   chart_img,
            "heatmap_img": heatmap_img,
        })
    return loaded #json item with keys: "imgname", "query", "label", "is_numerical", "saliency_map", "chart_img", "heatmap_img" (real images)



def run_inference(model, samples, train_samples, setting, use_saliency):
    results       = []

    for i, sample in enumerate(samples):
        # load from test json file
        imgname   = sample["imgname"]
        question  = sample["query"]
        gt_answer = sample["label"]
        is_numerical = sample["is_numerical"]
        is_year = sample["is_year"]
        saliency_map = sample["saliency_map"] if use_saliency else None

        chart_img   = Image.open(os.path.join(TEST_IMG_DIR, imgname)).convert("RGB")
        heatmap_img = Image.open(os.path.join(TEST_HEATMAP, saliency_map)).convert("RGB") if use_saliency  else None

        if setting == "zeroshot":
            prompt = build_prompt_zeroshot(question, chart_img, heatmap_img if use_saliency else None)
            predicted_answer = model.generate(prompt)

        elif setting == "fewshot":
            raw_examples = retrieve_examples(train_samples, sample)
            examples     = load_example_images(raw_examples, TRAIN_IMG_DIR, TRAIN_HEATMAP)
            prompt = build_prompt_fewshot(question, examples, chart_img, heatmap_img if use_saliency else None)
            print(f"heatmap for this question: {saliency_map}")
            print(f"Prompt for {imgname}:\n{prompt}\n")
            print("-----------------------------------------")
            predicted_answer = model.generate(prompt)

        results.append({
            "imgname":      imgname,
            "saliency_map": saliency_map if use_saliency else None,
            "question":     question,
            "gt_answer":    gt_answer,
            "pred_answer":  predicted_answer,
            "is_numerical": is_numerical,
            "is_year": is_year
            })

        if (i + 1) % 10 == 0:
            print(f"----------{i+1}/{len(samples)} processed.----------")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava15", "chartr1", "internvl", "qwen3vl", "bespokeminchart"], default="llava15")
    parser.add_argument("--setting", choices=["zeroshot", "fewshot"],  default="zeroshot")
    parser.add_argument("--use_saliency", action="store_true")
    parser.add_argument("--max_samples",  type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    saliency_tag = "with_saliency" if args.use_saliency else "no_saliency"
    output_path  = f"./random/{args.model}_{args.setting}_{saliency_tag}.json"

    with open(TEST_JSON, "r") as f:
        samples = json.load(f)[:args.max_samples] # load test samples, samples[0]["imgname"] = "1.png"

    train_samples = []
    if args.setting == "fewshot":
        with open(TRAIN_JSON, "r") as f:
            train_samples = json.load(f)[:100] # load train samples for few-shot retrieval

    model   = load_model(args.model)
    results = run_inference(model, samples, train_samples, args.setting, args.use_saliency)

    Path("./random").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# python inference_random.py --model qwen3vl  --setting fewshot --use_saliency
