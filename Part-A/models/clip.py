from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import TransformerBlock, VisionTransformer


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 40,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Token sequence length {seq_len} exceeds max_len={self.max_len}")

        x = self.token_embed(token_ids)
        x = x + self.pos_embed[:, :seq_len, :]

        attn_mask = self._build_causal_mask(seq_len=seq_len, device=token_ids.device)
        for blk in self.blocks:
            x = blk(x, mask=attn_mask)

        x = self.norm(x)

        # Last non-padding token (EOS effectively) per sequence.
        valid_counts = (token_ids != self.pad_id).sum(dim=1).clamp(min=1)
        last_indices = valid_counts - 1
        pooled = x[torch.arange(batch_size, device=token_ids.device), last_indices]
        return pooled


class CLIPModel(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionTransformer,
        text_encoder: TextTransformer,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.image_proj = nn.Linear(384, embed_dim)
        self.text_proj = nn.Linear(384, embed_dim)

        self.logit_scale = nn.Parameter(torch.tensor([torch.log(torch.tensor(1 / 0.07))], dtype=torch.float32))

    def encode_image(self, images: torch.Tensor, use_gap: bool = False) -> torch.Tensor:
        cls, patches = self.vision_encoder(images)
        feat = patches.mean(dim=1) if use_gap else cls
        feat = self.image_proj(feat)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        text_feat = self.text_encoder(token_ids)
        text_feat = self.text_proj(text_feat)
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        return text_feat

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(token_ids)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        return image_emb, text_emb, logit_scale


def clip_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    valid_pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric CLIP loss with optional valid pair masking."""
    logits_i2t = logit_scale * image_emb @ text_emb.t()
    logits_t2i = logits_i2t.t()

    if valid_pair_mask is not None:
        valid_pair_mask = valid_pair_mask.to(logits_i2t.device)
        invalid_mask = ~valid_pair_mask
        logits_i2t = logits_i2t.masked_fill(invalid_mask, -1e4)
        logits_t2i = logits_t2i.masked_fill(invalid_mask.t(), -1e4)

    targets = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = F.cross_entropy(logits_i2t, targets)
    loss_t = F.cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i + loss_t)
