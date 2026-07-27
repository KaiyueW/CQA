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

KNN_JSON        = "./fewshot_egs/twostep_dinov2_openai_knn_fewshot_examples.json" # the retrieval results for few-shot examples.

# Prompt builders 
# === CHANGE === new constant: the instruction used in step 2 to force a clean final answer
INST_USR_FINAL_LABEL = (
    "Based on your analysis above, now give the final answer only.\n"
    "Do not repeat your reasoning. Output only the final answer, nothing else.\n"
)


def build_prompt_zeroshot(question: str, chart_img, heatmap_img=None) -> list:
    if heatmap_img is not None:
        system = {
            "role": "system",
            "content": [
                {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given a chart image and a saliency map overlaid on the same chart.\n"
                "The saliency map represents human attention when answering the question, highlighting regions humans are likely to focus on.\n"
                "Use the saliency map as a helpful reference to identify potentially relevant regions, but verify all information directly from the chart image.\n"
                "Answer the question only based on the given images. Do not use external knowledge or assumptions.\n"
                }
            ]
        }
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "image", "image": heatmap_img},
                {"type": "text", "text":
                "The first image is the chart.\n"
                "The second image is the saliency map overlaid on the same chart, indicating regions likely relevant to the question.\n\n"
                f"Answer this question based on these two images: {question}\n\n"
                "First reason step by step about the answer based on the chart and saliency map.\n"
                }
            ]
        }
    else:
        system = {
            "role": "system",
            "content": [
                {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given a chart image.\n"
                "Answer the question only based on the given image. Do not use external knowledge or assumptions.\n"
                }
            ]
        }
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": chart_img},
                {"type": "text", "text":
                f"Answer this question based on the image: {question}\n\n"
                "First reason step by step about the answer based on the chart.\n"
                }
            ]
        }

    return [system, user]


# === CHANGE === 
def build_final_answer_prompt(step1_messages: list, analysis_text: str) -> list:
    assistant_turn = {
        "role": "assistant",
        "content": [{"type": "text", "text": analysis_text}]
    }
    final_user_turn = {
        "role": "user",
        "content": [{"type": "text", "text": INST_USR_FINAL_LABEL}]
    }
    return step1_messages + [assistant_turn, final_user_turn]

def build_prompt_fewshot(question: str, examples: list, chart_img, heatmap_img=None) -> list:

    # === CHANGE === system prompt no longer forces "final answer only" —
    # that constraint now lives in the step-2 extraction turn instead
    if heatmap_img is not None:
        system_content = [
            {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given several examples, each containing a chart, a saliency map, a question, and a final answer.\n"
                "The saliency map represents human attention when answering the question, highlighting regions humans are likely to focus on.\n"
                "Use the saliency map as a helpful reference to identify potentially relevant regions, but verify all information directly from the chart image.\n"
                "Learn the pattern from these examples and apply it to the final question.\n"
            }
        ]

    else:
        system_content = [
            {"type": "text", "text":
                "You are an expert chart question answering assistant.\n"
                "You will be given several examples, each containing a chart, a question, and a correct final answer.\n"
                "Learn the pattern from these examples and apply it to the final question.\n"
            }
        ]

    for i, ex in enumerate(examples, start=1):
        if heatmap_img is not None:
            system_content.append({"type": "image", "image": ex["chart_img"]})
            system_content.append({"type": "image", "image": ex["heatmap_img"]})
            system_content.append({"type": "text", "text":
                f"Example {i}:\n"
                "Given the chart and its saliency map, answer the following question.\n"
                f"Question: {ex['query']}\n"
                f"Answer: {ex['label']}\n"
            })
        else:
            system_content.append({"type": "image", "image": ex["chart_img"]})
            system_content.append({"type": "text", "text":
                f"Example {i}:\n"
                "Given the chart, answer the question.\n"
                f"Question: {ex['query']}\n"
                f"Answer: {ex['label']}\n"
            })

    prompt = [{"role": "system", "content": system_content}]

    # === CHANGE === user turn now asks the model to reason step by step,
    # mirroring build_prompt_zeroshot, instead of asking for the answer directly
    user_content = []
    if heatmap_img is not None:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "image", "image": heatmap_img})
        user_content.append({"type": "text", "text":
            "Given the chart and its saliency map, answer the following question.\n"
            f"Question: {question}\n\n"
            "Look at how the examples above approached similar questions, then reason step by step about this question.\n"
        })
    else:
        user_content.append({"type": "image", "image": chart_img})
        user_content.append({"type": "text", "text":
            "Given the chart, answer the question.\n"
            f"Question: {question}\n\n"
            "Look at how the examples above approached similar questions, then reason step by step about this question.\n"
        })

    prompt.append({"role": "user", "content": user_content})

    return prompt


# Few-shot example retrieval, here we simply retrieve other questions for the same chart.
def retrieve_examples(knn_json, current_sample, num_shot) -> list:
    key = current_sample["saliency_map"]
    candidates = knn_json.get(key)

    if candidates is None:
        raise ValueError(f"No KNN entry found in knn_fewshot_examples.json for key: {key}")

    chosen = candidates[:num_shot]
    #print(f"Samples: {chosen}")
    return chosen  # list of dicts with keys: "imgname", "query", "label"

def load_example_images(examples, img_dir, heatmap_dir):
    loaded  = []
    for ex in examples:
        chart_img   = Image.open(os.path.join(img_dir, ex["imgname"])).convert("RGB")
        # heatmap_img = Image.open(os.path.join(heatmap_dir, ex["saliency_map"])).convert("RGB")
        loaded.append({
            **ex,
            "chart_img":   chart_img,
            # "heatmap_img": heatmap_img,
        })
    return loaded #json item with keys: "imgname", "query", "label", "chart_img" (real images)



def run_inference(model, samples, knn_json, num_shot, setting, use_saliency):
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
            # === CHANGE === step 1: build the reasoning prompt and let the model think out loud
            step1_prompt = build_prompt_zeroshot(question, chart_img, heatmap_img if use_saliency else None)
            analysis_text = model.generate(step1_prompt, max_new_tokens=512)

            # === CHANGE === step 2: feed the analysis back as the assistant's own turn,
            # then ask a fresh user turn for the final answer only
            final_prompt = build_final_answer_prompt(step1_prompt, analysis_text)
            predicted_answer = model.generate(final_prompt, max_new_tokens=20)
            #print(f"for {saliency_map}, the prompt is {final_prompt}, the predicted answer is {predicted_answer}")
            #print("-----------------------------------------")

        elif setting == "fewshot":
            raw_examples = retrieve_examples(knn_json, sample, num_shot)
            examples     = load_example_images(raw_examples, TRAIN_IMG_DIR, TRAIN_HEATMAP)

            # === CHANGE === two-step generation, mirroring the zeroshot branch:
            # step 1 reasoning, step 2 final-answer extraction
            step1_prompt  = build_prompt_fewshot(question, examples, chart_img, heatmap_img if use_saliency else None)
            analysis_text = model.generate(step1_prompt, max_new_tokens=512)

            final_prompt      = build_final_answer_prompt(step1_prompt, analysis_text)
            predicted_answer  = model.generate(final_prompt, max_new_tokens=20)
            print(f"for {imgname}\nthe prompt is {analysis_text}\nthe predicted answer is {predicted_answer}\nthe ground truth answer is {gt_answer}")
            print("-----------------------------------------")

        results.append({
            "imgname":      imgname,
            "saliency_map": saliency_map if use_saliency else None,
            "question":     question,
            "gt_answer":    gt_answer,
            "pred_answer":  predicted_answer,
            "is_numerical": is_numerical,
            "is_year": is_year
            })

        if (i + 1) % 25 == 0:
            print(f"----------{i+1}/{len(samples)} processed.----------")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llava15", "chartr1", "internvl", "qwen3vl", "bespokeminchart"], default="llava15")
    parser.add_argument("--setting", choices=["zeroshot", "fewshot"],  default="zeroshot")
    parser.add_argument("--use_saliency", action="store_true")
    parser.add_argument("--max_samples",  type=int, default=None)
    parser.add_argument("--num_shot",     type=int, default=None)
    args = parser.parse_args()

    saliency_tag = "with_saliency" if args.use_saliency else "no_saliency"
    output_path  = f"./results/{args.model}_{args.setting}_{saliency_tag}_fewshot_reasoning_openai_{args.num_shot}.json"

    with open(TEST_JSON, "r") as f:
        samples = json.load(f)[:args.max_samples] # load test samples, samples[0]["imgname"] = "1.png"

    
    knn_json = {}
    if args.setting == "fewshot":
        with open(KNN_JSON, "r") as f:
            knn_json = json.load(f) # load the retrieval results for few-shot examples

    model   = load_model(args.model)
    results = run_inference(model, samples, knn_json, args.num_shot, args.setting, args.use_saliency)

    Path("./results").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# python inference_update.py --model internvl  --setting zeroshot --num_shot 3 --max_samples 3
# python evaluation.py --result_path ./results/qwen3vl_zeroshot_no_saliency_weighted_siglip_update_3.json