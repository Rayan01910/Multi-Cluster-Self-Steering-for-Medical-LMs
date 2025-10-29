import torch, torch.nn as nn

class ATSHead(nn.Module):
    """
    Predicts scalar temperature τ from hidden state h (last-layer last-token),
    then scales logits as z / exp(τ). We output τ_raw and use exp to ensure positivity.
    """
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1),
        )
    def forward(self, h):       # h: [B,d]
        orig_dtype = h.dtype
        ln = self.mlp[0]
        if isinstance(ln, nn.LayerNorm) and h.dtype != ln.weight.dtype:
            h = h.to(ln.weight.dtype)
        tau = torch.exp(self.mlp(h)).clamp(min=1e-3, max=50.0)  # [B,1]
        if tau.dtype != orig_dtype:
            tau = tau.to(orig_dtype)
        return tau