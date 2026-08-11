"""
Run:
  python prepare_data_saliency.py --annotations ../data/ChartQA_data/train/train_all_preprocessed.json --img_dir ../data/ChartQA_data/train/png --saliency_dir ../data/saliency_maps/ChartQA_train_all --split train --out_dir ./data
"""
import argparse
import json
import os
from pathlib import Path

# Matches the system/user prompt used in build_prompt_zeroshot(), so training
# and inference use identical instruction phrasing.
SYSTEM_PROMPT = ( 
    "You are an expert chart question answering assistant.\n"
    "You will be given a chart image and a saliency map overlaid on the same chart.\n"
    "The saliency map represents human attention when answering the question, highlighting regions humans are likely to focus on.\n"
    "Use the saliency map as a helpful reference to identify potentially relevant regions, but verify all information directly from the chart image.\n"
    "Answer the question only based on the given images. Do not use external knowledge or assumptions.\n"
    "Return ONLY the final answer. Do not include explanation or reasoning.\n"            
)


USER_TEMPLATE = (
    "<image>This is the chart.\n"
    "<image>This is the saliency map overlaid on the same chart, indicating regions likely relevant to the question.\n\n"
    "Answer this question based on these two images: {question}\n\n"
)


def load_records(annotations_path: str, img_dir: str, saliency_dir: str):
    with open(annotations_path) as f:
        raw = json.load(f)

    img_dir = Path(img_dir)
    saliency_dir = Path(saliency_dir)
    records = []
    n_missing = 0

    for r in raw:
        img_path = img_dir / r["imgname"]
        saliency_path = saliency_dir / r["saliency_map"]
        if not img_path.exists() or not saliency_path.exists():
            n_missing += 1
            continue
        records.append({
            "image_path": str(img_path),
            "saliency_path": str(saliency_path),
            "question": r["query"],
            "answer": r["label"],
        })

    if n_missing:
        print(f"WARNING: {n_missing} records skipped, chart image or saliency map not found in {img_dir} or {saliency_dir}")

    return records


def build_record(rec):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=rec["question"])},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["image_path"], rec["saliency_path"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="path to your train/test json file")
    parser.add_argument("--img_dir", required=True, help="folder containing chart images (imgname files)")
    parser.add_argument("--saliency_dir", required=True, help="folder containing saliency maps")
    parser.add_argument("--split", default="train", help="used only to name the output file")
    parser.add_argument("--out_dir", default="./data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.annotations, args.img_dir, args.saliency_dir)
    print(f"Loaded {len(records)} usable QA pairs")

    out_path = Path(args.out_dir) / f"{args.split}_all_saliency.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(build_record(rec)) + "\n")

    print(f"Wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()