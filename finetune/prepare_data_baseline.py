"""
Convert your ChartQA-style JSON into ms-swift SFT format for the BASELINE
fine-tune: chart image + question -> answer. No saliency map involved.

Input JSON is a list of records like:
  {
    "imgname": "41699051005347.png",
    "query": "How many food item is shown in the bar graph?",
    "label": "14",
    "is_numerical": true,
    "saliency_map": "41699051005347_Q0.png",
    "is_year": false
  }

We only use imgname/query/label here -- saliency_map is ignored for this run.

Run:
  python prepare_data_baseline.py --annotations ../data/ChartQA_data/train/train_all_preprocessed.json --img_dir ../data/ChartQA_data/train/png --split train --out_dir ./data

  python prepare_data_baseline.py \
      --annotations /path/to/test.json \
      --img_dir /path/to/chart/images \
      --split test \
      --out_dir ./data
"""
import argparse
import json
import os
from pathlib import Path

# Matches the system/user prompt used in build_prompt_zeroshot(), so training
# and inference use identical instruction phrasing.
SYSTEM_PROMPT = (
    "You are an expert chart question answering assistant.\n"
    "You will be given a chart image.\n"
    "Answer the question only based on the given images. Do not use external knowledge or assumptions.\n"
    "Return ONLY the final answer. Do not include explanation or reasoning.\n"
)

USER_TEMPLATE = "<image>Answer this question based on the image: {question}\n\n"


def load_records(annotations_path: str, img_dir: str):
    with open(annotations_path) as f:
        raw = json.load(f)

    img_dir = Path(img_dir)
    records = []
    n_missing = 0

    for r in raw:
        img_path = img_dir / r["imgname"]
        if not img_path.exists():
            n_missing += 1
            continue
        records.append({
            "image_path": str(img_path),
            "question": r["query"],
            "answer": r["label"],
        })
        print(f"Loaded record: {img_path} | Q: {r['query']} | A: {r['label']}")

    if n_missing:
        print(f"WARNING: {n_missing} records skipped, chart image not found in {img_dir}")

    return records


def build_record(rec):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=rec["question"])},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["image_path"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="path to your train/test json file")
    parser.add_argument("--img_dir", required=True, help="folder containing chart images (imgname files)")
    parser.add_argument("--split", default="train", help="used only to name the output file")
    parser.add_argument("--out_dir", default="./data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.annotations, args.img_dir)
    print(f"Loaded {len(records)} usable QA pairs")

    out_path = Path(args.out_dir) / f"{args.split}_all_no_saliency.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(build_record(rec)) + "\n")

    print(f"Wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()