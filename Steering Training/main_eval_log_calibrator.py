import numpy as np
import torch
from tqdm import tqdm
from config import cfg
from data_medqa import load_medqa
from models import CausalLMWrapper
from steering import softmax_cosine_weights
from projection import LinearProjection
from calibrator import ConfidenceCalibrator
from metrics import brier_score, ece, auroc

def build_prompt(question: str, options: list[str]) -> str:
    fmt = (
        "You are a clinical reasoning assistant.\n"
        "Question:\n{q}\n\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
        "Answer with just the letter (A/B/C/D)."
    )
    return fmt.format(q=question, a=options[0], b=options[1], c=options[2], d=options[3])

def main():
    torch.set_grad_enabled(False)
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

    # -------- Train calibrator on train set internal signals --------
    X_train, y_train = [], []
    for ex in tqdm(train_ds, desc="Calibrator features (train)"):
        qtxt = ex[cfg.text_field]
        opts = [ex[f] for f in cfg.option_fields]
        y = int(ex[cfg.label_field])
        prompt = build_prompt(qtxt, opts)

        # Per-option features
        opt_hidden = []
        opt_logp_steered = []
        option_feats = []

        for opt in opts:
            h = lm.option_hidden(prompt, f"\nAnswer: {opt}\n")
            p = P(h)
            w, sims = softmax_cosine_weights(p, centers_t, temp=cfg.cosine_temp)
            s = (w.unsqueeze(1) * centers_t).sum(dim=0)  # (D,)
            h_prime = h + cfg.alpha * s

            # Re-score using steered last-state by replacing h with h'
            # (Approximation: use original logprob; add a small bonus term from sims⋅w to reflect steering.
            # If you want exact re-forwarding with hooks, add a model hook to replace last hidden; kept simple here.)
            logp = lm.option_logprob(prompt, f"\nAnswer: {opt}\n")
            bonus = float((w * sims).sum().item())  # alignment bonus
            opt_hidden.append((h, p, w, s))
            opt_logp_steered.append(logp + 0.02 * bonus)  # small weight for alignment bonus

            # store temp features per option; final features need top option
            maxw = float(w.max().item())
            marginw = float((w.max() - torch.topk(w, 2).values[1]).item())
            cos_ps = float(
                torch.nn.functional.cosine_similarity(
                    p.unsqueeze(0), s.unsqueeze(0), dim=-1
                ).item()
            )
            norm_s = float(s.norm().item())
            option_feats.append((maxw, marginw, cos_ps, norm_s))

        # choose predicted index
        j_star = int(np.argmax(opt_logp_steered))
        # compute logprob margin vs runner-up
        sorted_lp = sorted(opt_logp_steered, reverse=True)
        delta_lp = float(sorted_lp[0] - sorted_lp[1])
        maxw, marginw, cos_ps, norm_s = option_feats[j_star]
        feat = [maxw, marginw, cos_ps, norm_s, delta_lp]
        X_train.append(feat)
        y_train.append(1 if j_star == y else 0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    calib = ConfidenceCalibrator()
    calib.fit(X_train, y_train)

    # -------- Evaluate on val & test --------
    def run_split(split_ds, split_name: str):
        X, y_true, y_hat = [], [], []
        for ex in tqdm(split_ds, desc=f"Eval {split_name}"):
            qtxt = ex[cfg.text_field]
            opts = [ex[f] for f in cfg.option_fields]
            y = int(ex[cfg.label_field])
            prompt = build_prompt(qtxt, opts)

            opt_logp_steered = []
            option_feats = []
            for opt in opts:
                h = lm.option_hidden(prompt, f"\nAnswer: {opt}\n")
                p = P(h)
                w, sims = softmax_cosine_weights(p, centers_t, temp=cfg.cosine_temp)
                s = (w.unsqueeze(1) * centers_t).sum(dim=0)
                logp = lm.option_logprob(prompt, f"\nAnswer: {opt}\n")
                bonus = float((w * sims).sum().item())
                opt_logp_steered.append(logp + 0.02 * bonus)

                maxw = float(w.max().item())
                marginw = float((w.max() - torch.topk(w, 2).values[1]).item())
                cos_ps = float(torch.nn.functional.cosine_similarity(p.unsqueeze(0), s.unsqueeze(0), dim=-1).item())
                norm_s = float(s.norm().item())
                option_feats.append((maxw, marginw, cos_ps, norm_s))

            j_star = int(np.argmax(opt_logp_steered))
            sorted_lp = sorted(opt_logp_steered, reverse=True)
            delta_lp = float(sorted_lp[0] - sorted_lp[1])
            maxw, marginw, cos_ps, norm_s = option_feats[j_star]
            feat = np.array([maxw, marginw, cos_ps, norm_s, delta_lp], dtype=np.float32)

            prob = float(calib.predict_proba(feat.reshape(1, -1))[0])

            X.append(feat)
            y_hat.append(prob)
            y_true.append(1 if j_star == y else 0)

        X = np.vstack(X)
        y_hat = np.array(y_hat, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int32)
        acc = float(( (y_hat >= 0.5).astype(int) == y_true ).mean())
        bs = brier_score(y_hat, y_true)
        e = ece(y_hat, y_true, bins=cfg.ece_bins)
        r = auroc(y_hat, y_true)
        print(f"[{split_name}] Acc={acc:.4f}  AUROC={r:.4f}  Brier={bs:.4f}  ECE={e:.4f}")
        return acc, r, bs, e

    print("Validation:")
    run_split(val_ds, "val")
    print("Test:")
    run_split(test_ds, "test")

if __name__ == "__main__":
    main()
