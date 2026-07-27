import json
import re
import argparse
from sympy import sympify, Number
from sympy.core.sympify import SympifyError
import editdistance

def extract_number(text: str) -> float:
    first_line = text.split('\n')[0]
    # print(first_line)
    if '=' in first_line: 
        text = first_line.split('=')[-1].strip() # if the answer is in format like "Answer = 42", extract the part after "="
    else:
        text = first_line.strip() # clean whitespace

    try:
        cleaned = text.replace(',', '').strip('%')
        result = sympify(cleaned)
    # === CHANGE ===
        if not hasattr(result, "is_number"):
            return None
    # === END CHANGE ===
        if result.is_number:
            return float(result)   
        return None
    except (SympifyError, TypeError):
        pass

    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", text)
    if match:
        num_str = match.group(0).replace(',', '')
        try:
            return float(num_str.strip('%'))
        except ValueError:
            return None
    return None

def build_gt_candidates(gt: float) -> list:
    candidates = [gt]

    if 0 < abs(gt) < 5: # gt = 0.19
        candidates.append(gt * 100)
    elif abs(gt) < 100: # gt = 19.22
        candidates.append(gt / 100)  
    return candidates

def within_tolerance(gt_num: float, pred_num: float) -> bool:
    return abs(gt_num - pred_num) / (abs(gt_num) + 1e-9) <= 0.05

def extract_text(pred: str) -> str:
    pred = re.sub(r'<[^>]+>', '', pred).strip() # delete tags like </p>
    first_line = pred.split('\n')[0].strip()
    return first_line
    
def is_textual_correct(gt, pred) -> bool:
    gt_clean = str(gt).strip().lower()
    pred_clean = str(pred).strip().lower()

    if gt_clean == pred_clean.rstrip(".,"):
        return True
    
    if gt_clean in pred_clean:
        return True

    if pred_clean in gt_clean:
        return True
    
    if editdistance.eval(gt_clean, pred_clean.rstrip('.,')) <= 2:
        return True

    return False


def is_correct(gt, pred, is_numerical, is_year):
    if is_numerical:
        gt_num = float(gt.strip().replace(',', '').strip('%')) # no %
        pred_num = extract_number(pred) # no %
        # print(f"-----Extracted numbers: GT={gt_num}, Pred={pred_num}")
        if gt_num is None or pred_num is None:
            return False
        if is_year: # for year answers, allow absolute error of 1 year
            return abs(gt_num - pred_num) <= 1
        else: # allow 5% relative error for non-year numerical answers
            candidates = build_gt_candidates(gt_num)
            for c in candidates:
                if within_tolerance(c, pred_num):
                    return True
            return False

    else: 
        pred_text = extract_text(pred)
        # (f"-----Extracted text: GT={gt}, Pred={pred_text}")
        return is_textual_correct(gt, pred_text)


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
        is_year = result["is_year"]
        i = i+1
        # print("========================================================================")
        #print(f"-----Result {i}: gt_answer={gt}, pred_answer={pred}, is_numerical={is_numerical}, is_year={is_year}")
        
        correct = is_correct(gt, pred, is_numerical, is_year)
        # print(f"-----Correct: {correct}")
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



