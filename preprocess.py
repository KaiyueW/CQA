import json
import os
from collections import defaultdict

def is_numerical(label: str) -> bool:
    try:
        float(label.replace(',', '')) # whether can convert into float
        return True
    except ValueError:
        return False


JSON_PATH   = "./ChartQA_data/test/test_human.json"
OUTPUT_PATH = "./ChartQA_data/test/test_human_preprocessed.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

counter = defaultdict(int)
for item in data:
    stem  = os.path.splitext(item["imgname"])[0]  # "416.png" → "416"
    q_idx = counter[stem]
    counter[stem] += 1

    item["is_numerical"]  = is_numerical(item["label"])
    item["saliency_map"]  = f"{stem}_Q{q_idx}.png"

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Done! Saved to {OUTPUT_PATH}")
print(f"Total: {len(data)} items")
print(f"Numerical: {sum(1 for item in data if item['is_numerical'])}")
print(f"Non-numerical: {sum(1 for item in data if not item['is_numerical'])}")