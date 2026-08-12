import torch
import torch.nn as nn

class ReverseBNProjector(nn.Module):

    def __init__(self,in_dim=384,hidden_dim=1536,out_dim=2560) -> None:
        super().__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        # Input normalisation, applied before the first linear layer.
        # Keeps ViT feature scale compatible with Qwen's residual stream.
        self.norm=nn.LayerNorm(in_dim)
        self.fc1=nn.Linear(in_dim, hidden_dim, bias=True)# Expansion layer:384 → 1536
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden_dim, out_dim, bias=True)# Output layer:1536 → 2560
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.fc1, self.fc2):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, patch_tokens:torch.Tensor) -> torch.Tensor:
        x=self.norm(patch_tokens) # (B, N, 384)
        x=self.fc1(x)# (B, N, 1536)
        x=self.act(x)# (B, N, 1536)
        x=self.fc2(x)# (B, N, 2560)
        return x