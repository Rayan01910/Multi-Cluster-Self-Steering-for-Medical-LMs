import csv
import numpy as np
from eval_metrics import get_brier, get_ece, get_auroc


def recalculate_metrics():
    correct_list = []
    confidence_list = []
    
    with open("results.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert bool to int
            correct_val = row["correct"].strip().lower()
            if correct_val == "true":
                correct_list.append(1)
            elif correct_val == "false":
                correct_list.append(0)
            else:
                raise ValueError(f"Unexpected correct value: {row['correct']}")
            confidence_list.append(float(row["confidence"]))
    
    correct_arr = np.array(correct_list)
    confidence_arr = np.array(confidence_list)
    
    mean_auroc = get_auroc(correct_arr, confidence_arr)
    mean_ece = get_ece(correct_arr, confidence_arr)
    mean_brier = get_brier(correct_arr, confidence_arr)
    
    print(f"AUROC: {mean_auroc}")
    print(f"ECE: {mean_ece}")
    print(f"Brier: {mean_brier}")
