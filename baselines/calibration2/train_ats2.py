import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.data2.medqa_dataset2 import MedQADataset, LETTER
from baselines.data2.prompt_builder2 import build_prompt
from baselines.model2.loader2 import load_model
from baselines.model2.hooks2 import last_hidden_last_token
from baselines.calibration2.ats_head2 import ATSHead
from baselines.steering2.config2 import DEVICE, TARGET_LAYER, ATS_PATH, BATCH_SIZE

def selective_loss(q_logits, y_idx, alpha=0.5):
    # q_logits: [B, 4] calibrated logits (after scaling), y_idx: [B]
    ce = nn.CrossEntropyLoss(reduction="none")
    base = ce(q_logits, y_idx)  # per-sample
    with torch.no_grad():
        correct = (q_logits.argmax(dim=-1) == y_idx).float()
    uni = -torch.log_softmax(q_logits, dim=-1).mean(dim=-1)  # uniform target
    return torch.where(correct>0, (1-alpha)*base, alpha*uni).mean()

@torch.no_grad()
def hidden_and_logits(tok, model, batch):
    H=[]; Z=[]
    stems = batch["stem"]
    choices_batch = batch["choices"]

    # ``DataLoader``'s default collate_fn stacks list values by column, so a
    # batch of ``["A", "B", "C", "D"]`` options becomes a list with four
    # entries where each entry contains one option per example. For example, a
    # batch size of 2 yields ``[("A", "A"), ("B", "B"), ("C", "C"),
    # ("D", "D")]``. We need to transpose this structure back to the expected
    # per-example layout before building prompts.
    def _pull_option(container, key, idx):
        if key in container:
            return container[key]
        fallback = str(idx)
        if fallback in container:
            return container[fallback]
        raise KeyError(f"Missing option '{key}' in collated batch")

    if isinstance(choices_batch, dict):
        # Default PyTorch collation converts a list[dict] into dict[list].
        choices_batch = list(
            zip(*(_pull_option(choices_batch, key, idx) for idx, key in enumerate(LETTER)))
        )
    elif (
            isinstance(choices_batch, (list, tuple))
            and choices_batch
            and isinstance(choices_batch[0], dict)
    ):
        # ``MedQADataset`` may have already returned list[dict[str,str]].
        choices_batch = [
            tuple(_pull_option(choice, key, idx) for idx, key in enumerate(LETTER))
            for choice in choices_batch
        ]
    elif (
            isinstance(choices_batch, (list, tuple))
            and len(choices_batch) == len(LETTER)
            and isinstance(choices_batch[0], (list, tuple))
    ):
        choices_batch = list(zip(*choices_batch))

    for stem, choices in zip(stems, choices_batch):
        prompt = build_prompt(stem, list(choices))
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        out = model(**inputs)
        h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0)     # [d]
        logits = model.lm_head(out.hidden_states[-1][:, -1, :])      # [1,V]
        ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
        z = logits[:, ids].squeeze(0)                                # [4]
        H.append(h.cpu()); Z.append(z.cpu())
    return torch.stack(H), torch.stack(Z)

def train(split="validation"):
    tok, model = load_model()
    ds = MedQADataset(split=split)  # use a held-out slice as calibration set
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # infer hidden size
    h0, _ = hidden_and_logits(tok, model, next(iter(loader)))
    dim = h0.shape[-1]
    head = ATSHead(dim).to(DEVICE)

    opt = optim.AdamW(head.parameters(), lr=5e-5, betas=(0.9,0.999), weight_decay=0.0)

    for epoch in range(2):
        pbar = tqdm(loader, desc=f"ATS epoch {epoch}")
        for batch in pbar:
            H, Z = hidden_and_logits(tok, model, batch)
            H = H.to(DEVICE); Z = Z.to(DEVICE)
            y = torch.tensor(batch["label"], dtype=torch.long, device=DEVICE)

            tau = head(H)                    # [B,1]
            Zcal = Z / tau                   # temperature scaling
            loss = selective_loss(Zcal, y, alpha=0.5)

            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    torch.save(head.state_dict(), ATS_PATH)
    return head
