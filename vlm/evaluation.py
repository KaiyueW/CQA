import json
import re
import argparse

def extract_number(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", text)
    if match:
        num_str = match.group(0).replace(',', '')
        try:
            return float(num_str.strip('%'))
        except ValueError:
            return None
    return None
    
def is_correct(gt, pred, is_numerical):
    if is_numerical:
        gt_num = extract_number(gt)
        pred_num = extract_number(pred)
        print(f"-----Extracted numbers: GT={gt_num}, Pred={pred_num}")
        if gt_num is None or pred_num is None:
            return False
        return abs(gt_num - pred_num) / (abs(gt_num) + 1e-9) <= 0.05
    else:
        return str(gt).strip().lower() == str(pred).strip().lower()


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
    i = 0

    for result in results:
        gt = result["gt_answer"]
        pred = result["pred_answer"]
        is_numerical = result["is_numerical"]
        i = i+1
        print("========================================================================")
        print(f"-----Result {i}: gt_answer={gt}, pred_answer={pred}, is_numerical={is_numerical}")

        correct = is_correct(gt, pred, is_numerical)
        print(f"-----Correct: {correct}")
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