import numpy as np

def brier_score(q: np.ndarray, y: np.ndarray) -> float:
    return np.mean((q - y) ** 2)

def ece(q: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    # q in [0,1]
    edges = np.linspace(0, 1, bins + 1)
    ece_val = 0.0
    N = len(q)
    for i in range(bins):
        mask = (q >= edges[i]) & (q < edges[i+1]) if i < bins - 1 else (q >= edges[i]) & (q <= edges[i+1])
        if mask.sum() == 0: continue
        conf = q[mask].mean()
        acc = y[mask].mean()
        ece_val += (mask.sum() / N) * abs(acc - conf)
    return ece_val

def auroc(q: np.ndarray, y: np.ndarray) -> float:
    # simple O(N log N) AUROC
    order = np.argsort(-q)  # descending scores
    y_sorted = y[order]
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.5
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / pos
    fpr = fps / neg
    # trapezoidal integration
    area = np.trapz(tpr, fpr)
    return float(area)
