import numpy as np
import torch
from sklearn.cluster import KMeans
from config import cfg

def compute_sample_steering(h_correct: torch.Tensor, h_incorrect: list[torch.Tensor]) -> np.ndarray:
    # v = h_pos - mean(h_neg)
    h_neg = torch.stack(h_incorrect, dim=0).mean(dim=0)
    v = (h_correct - h_neg).detach().float().cpu().numpy()
    return v

def kmeans_dictionary(V: np.ndarray) -> tuple[np.ndarray, KMeans]:
    # V: (N, D)
    km = KMeans(n_clusters=cfg.num_clusters, random_state=cfg.kmeans_seed, n_init="auto")
    km.fit(V)
    centers = km.cluster_centers_.astype(np.float32)  # (K, D)
    # L2-normalize centers for cosine
    norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    centers = centers / norms
    return centers, km

def softmax_cosine_weights(p_vec: torch.Tensor, centers: torch.Tensor, temp: float) -> torch.Tensor:
    # p_vec: (D,), centers: (K, D)
    p = p_vec / (p_vec.norm() + 1e-8)
    sims = torch.matmul(centers, p)  # (K,)
    w = torch.softmax(temp * sims, dim=0)
    return w, sims
