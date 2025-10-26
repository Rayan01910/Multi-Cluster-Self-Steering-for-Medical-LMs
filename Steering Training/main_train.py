import numpy as np
import torch
from tqdm import tqdm
from config import cfg
from data_medqa import load_medqa
from models import CausalLMWrapper
from steering import compute_sample_steering, kmeans_dictionary
from projection import train_projection
from calibrator import ConfidenceCalibrator

def build_prompt(question: str, options: list[str]) -> str:
    # Minimal MCQ format (change as you like)
    fmt = (
        "You are a clinical reasoning assistant.\n"
        "Question:\n{q}\n\nOptions:\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
        "Answer with just the letter (A/B/C/D)."
    )
    return fmt.format(q=question, a=options[0], b=options[1], c=options[2], d=options[3])

def main():
    torch.set_grad_enabled(False)
    ds = load_medqa()
    train_ds = ds["train"]

    lm = CausalLMWrapper()
    # Collect steering vectors and training pairs for projection P
    H_in = []
    V_tar = []

    # Also collect cluster inputs (same V list)
    Vs = []

    for ex in tqdm(train_ds, desc="Collecting steering pairs"):
        q = ex[cfg.text_field]
        opts = [ex[f] for f in cfg.option_fields]
        y = int(ex[cfg.label_field])

        prompt = build_prompt(q, opts)

        # per-option h and keep tensors
        h_list = []
        for opt in opts:
            h = lm.option_hidden(prompt, f"\nAnswer: {opt}\n")
            h_list.append(h)
        h_correct = h_list[y]
        h_incorrect = [h_list[i] for i in range(4) if i != y]

        v = compute_sample_steering(h_correct, h_incorrect)  # np.ndarray (D,)
        Vs.append(v)
        H_in.append(h_correct.detach().float().cpu().numpy())
        V_tar.append(v)

    V = np.stack(Vs, axis=0)                # (N, D)
    H_in_np = np.stack(H_in, axis=0)
    V_tar_np = np.stack(V_tar, axis=0)

    # K-means dictionary
    centers_np, km = kmeans_dictionary(V)
    np.save("steering_dictionary.npy", centers_np)

    # Train projection P to map h_correct -> v
    H_t = torch.from_numpy(H_in_np).to(cfg.device)
    V_t = torch.from_numpy(V_tar_np).to(cfg.device)
    P = train_projection(H_t, V_t, dim=H_t.shape[1])
    torch.save(P.state_dict(), "projection.pt")

    print("Dictionary and projection trained & saved.")

if __name__ == "__main__":
    main()
