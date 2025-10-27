# main_eval.py  (TS-softmax version; no logistic regression calibrator)
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from typing import List, Tuple
from config import cfg
from data_medqa import load_medqa
from models import CausalLMWrapper
from steering import softmax_cosine_weights
from projection import LinearProjection
from metrics import brier_score, ece, auroc

def build_prompt(question: str, options: list[str]) -> str:
    # Keep this consistent with training prompt (system prompt is integrated there).
    # If you also integrated system prompt during eval, mirror it here the same way.
    fmt = (
        "You are a clinical reasoning assistant.\n"
        "Question:\n{q}\n\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
        "Answer with just the letter (A/B/C/D)."
    )
    return fmt.format(q=question, a=options[0], b=options[1], c=options[2], d=options[3])

# ---------- Temperature Scaling (scalar T) ----------
class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # Optimize log_T to keep T>0
        self.log_T = nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

    def temperature(self) -> float:
        return float(torch.exp(self.log_T).detach().cpu().item())

def fit_temperature(logit_list: List[np.ndarray], label_list: List[int], max_iter: int = 500) -> float:
    """
    Fit scalar temperature T by minimizing NLL on validation set.
    logit_list: list of shape-(4,) arrays per example (the uncalibrated scores).
    label_list: gold indices (0..3).
    """
    device = cfg.device
    L = torch.from_numpy(np.stack(logit_list, axis=0)).to(device)  # (N,4)
    y = torch.tensor(label_list, dtype=torch.long, device=device)   # (N,)

    ts = TemperatureScaler(init_T=1.0).to(device)
    opt = optim.LBFGS(ts.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        scaled = ts(L)
        loss = nll(scaled, y)
        loss.backward()
        return loss

    opt.step(closure)
    return ts.temperature()

def softmax_probs(z: np.ndarray, T: float) -> np.ndarray:
    zt = z / T
    zt = zt - zt.max()  # stability
    exp = np.exp(zt)
    return exp / exp.sum()

def main():
    torch.set_grad_enabled(False)

    # Load dictionary + projection
    centers = np.load("steering_dictionary.npy")  # (K, D)
    centers_t = torch.from_numpy(centers).to(cfg.device)
    lm = CausalLMWrapper()
    P = LinearProjection(lm.hidden_size).to(cfg.device)
    P.load_state_dict(torch.load("projection.pt", map_location=cfg.device))
    P.eval()

    ds = load_medqa()
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    test_ds  = ds["test"]

    # --------- Helper to compute per-item uncalibrated scores (z) with steering ---------
    def item_scores(question: str, options: List[str]) -> Tuple[np.ndarray, List[float]]:
        """
        Returns:
          z: np.array shape (4,) of uncalibrated scores (base logprob + small alignment bonus)
          aux: optional debug list of bonuses per option
        """
        prompt = build_prompt(question, options)
        z = []
        bonuses = []
        for opt in options:
            h = lm.option_hidden(prompt, f"\nAnswer: {opt}\n")
            p = P(h)
            w, sims = softmax_cosine_weights(p, centers_t, temp=cfg.cosine_temp)
            s = (w.unsqueeze(1) * centers_t).sum(dim=0)
            # base score: sum logprob of option tokens (acts like a logit proxy)
            base = lm.option_logprob(prompt, f"\nAnswer: {opt}\n")
            # small alignment bonus to reflect steering alignment (kept same as before)
            bonus = float((w * sims).sum().item()) * 0.02
            bonuses.append(bonus)
            z.append(base + bonus)
        return np.array(z, dtype=np.float32), bonuses

    # --------- Collect validation logits/labels for temperature fitting ---------
    val_logits, val_labels = [], []
    for ex in tqdm(val_ds, desc="Collect val logits for T-fitting"):
        qtxt = ex[cfg.text_field]
        opts = [ex[f] for f in cfg.option_fields]
        y = int(ex[cfg.label_field])
        z, _ = item_scores(qtxt, opts)
        val_logits.append(z)
        val_labels.append(y)

    T = fit_temperature(val_logits, val_labels)
    print(f"[TS] Fitted temperature T = {T:.4f} (Wang et al., 2025 baseline)")

    # --------- Evaluate on splits using decision-level metrics (max prob) ---------
    def run_split(split_ds, split_name: str):
        y_true, conf, pred = [], [], []
        for ex in tqdm(split_ds, desc=f"Eval {split_name} (TS-softmax)"):
            qtxt = ex[cfg.text_field]
            opts = [ex[f] for f in cfg.option_fields]
            y = int(ex[cfg.label_field])

            z, _ = item_scores(qtxt, opts)
            p = softmax_probs(z, T)
            j_hat = int(np.argmax(p))
            y_true.append(1 if j_hat == y else 0)
            conf.append(float(p[j_hat]))
            pred.append(j_hat)

        y_true = np.array(y_true, dtype=np.int32)          # correctness (0/1)
        conf   = np.array(conf, dtype=np.float32)          # decision-level confidence (max p)

        acc = float((y_true == 1).mean())
        r   = auroc(conf, y_true)                          # decision-level AUROC
        bs  = brier_score(conf, y_true)                    # decision-level Brier
        e   = ece(conf, y_true, bins=cfg.ece_bins)         # decision-level ECE

        print(f"[{split_name}] Acc={acc:.4f}  AUROC={r:.4f}  Brier={bs:.4f}  ECE={e:.4f}")
        return acc, r, bs, e

    print("Validation:")
    run_split(val_ds, "val")

    print("Test:")
    run_split(test_ds, "test")

if __name__ == "__main__":
    main()
