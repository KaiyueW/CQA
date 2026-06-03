import json
import re
import argparse

def is_correct(gt, pred, is_numerical):
    if is_numerical:
        gt_num = extract_number(gt)
        pred_num = extract_number(pred)
        if gt_num is None or pred_num is None:
            return False
        return 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.result_path, "r") as f:
        results = json.load(f)
    
    total_correct = 0
    total = len(results)
    num_correct = 0
    num_total = 0
    text_correct = 0
    text_total = 0

    for result in results:
        gt = result["gt_answer"]
        pred = result["pred_answer"]
        is_numerical = result["is_numerical"]

        correct = is_correct(gt, pred, is_numerical)
        total_correct += correct

        if is_numerical:
            num_correct += correct
            num_total += 1
        else:
            text_correct += correct
            text_total += 1

    print(f"Total accuracy: {total_correct / total * 100:.2f}%")
    print(f"Numerical accuracy: {num_correct / num_total * 100:.2f}%")
    print(f"Text accuracy: {text_correct / text_total * 100:.2f}%")

if __name__ == "__main__":
    main()