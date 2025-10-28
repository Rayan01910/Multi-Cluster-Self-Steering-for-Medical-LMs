
import hashlib  
import numpy as np 
from sklearn.metrics import roc_curve, auc
import re
import string


def get_brier(correctness, confidence): #an array of responses' correctness and an array for the responses' confidence
    brier_score = np.mean((confidence - correctness) ** 2) #calculates the mean of the (confidence - correctness) ^2
    return brier_score

def get_ece(correctness,confidence): #same arrays passed in as previous func
    # Calculate ECE using 10 bins, including 0 and 1
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Ensure 0 and 1 are included in their own bins
    bin_edges[0] = -np.inf  # Include 0 in first bin
    bin_edges[-1] = np.inf  # Include 1 in last bin
    bin_indices = np.digitize(confidence, bin_edges) - 1
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(confidence[mask])
            bin_acc = np.mean(correctness[mask])
            bin_weight = np.sum(mask) / len(confidence)
            ece += bin_weight * np.abs(bin_conf - bin_acc)
    return ece

def get_auroc(correctness,confidence): #same arrays passed in as previous func
    fpr, tpr, _ = roc_curve(correctness, confidence)
    auroc = auc(fpr, tpr)
    return auroc



