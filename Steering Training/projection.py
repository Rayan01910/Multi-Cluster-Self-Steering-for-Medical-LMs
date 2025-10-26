import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from config import cfg

class ProjDataset(Dataset):
    def __init__(self, H_in: torch.Tensor, V_target: torch.Tensor):
        self.H = H_in  # (N, D)
        self.V = V_target  # (N, D)
    def __len__(self): return self.H.shape[0]
    def __getitem__(self, idx): return self.H[idx], self.V[idx]

class LinearProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.P.weight)  # start as identity

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.P(h)

def cosine_embed_loss(x, y):
    x = nn.functional.normalize(x, dim=-1)
    y = nn.functional.normalize(y, dim=-1)
    # maximize cosine => minimize (1 - cos)
    return (1.0 - (x * y).sum(dim=-1)).mean()

def train_projection(H_in: torch.Tensor, V_target: torch.Tensor, dim: int) -> LinearProjection:
    model = LinearProjection(dim).to(H_in.device)
    ds = ProjDataset(H_in, V_target)
    dl = DataLoader(ds, batch_size=cfg.proj_batch_size, shuffle=True, drop_last=False)
    opt = optim.AdamW(model.parameters(), lr=cfg.proj_lr, weight_decay=cfg.proj_weight_decay)
    model.train()
    for _ in range(cfg.proj_epochs):
        for h, v in dl:
            opt.zero_grad(set_to_none=True)
            h = h.to(H_in.device)
            v = v.to(H_in.device)
            pred = model(h)
            loss = cosine_embed_loss(pred, v)
            loss.backward()
            opt.step()
    return model
