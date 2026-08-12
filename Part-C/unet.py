import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils






class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, sin_dim: int, out_dim: int):
        super().__init__()
        self.sin = SinusoidalPosEmb(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))



class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1    = nn.GroupNorm(min(32, in_ch),  in_ch,  eps=1e-6)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(min(32, out_ch), out_ch, eps=1e-6)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act      = nn.SiLU()
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.norm     = nn.GroupNorm(min(32, dim), dim, eps=1e-6)
        self.to_qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.to_out   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        return x + self.to_out(out).permute(0, 2, 1).view(B, C, H, W)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int):
        super().__init__()
        assert query_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = query_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.norm     = nn.GroupNorm(min(32, query_dim), query_dim, eps=1e-6)
        self.to_q     = nn.Linear(query_dim,  query_dim, bias=False)
        self.to_k     = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v     = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out   = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        q = self.to_q(h).view(B, H*W, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        return x + self.to_out(out).permute(0, 2, 1).view(B, C, H, W)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim: int, context_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            n_heads,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attended, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attended

        h = self.norm2(x)
        attended, _ = self.cross_attn(h, context, context, need_weights=False)
        x = x + attended

        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):

    def __init__(self, ch: int, context_dim: int, n_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch, eps=1e-6)
        self.proj_in = nn.Conv2d(ch, ch, 1)
        self.block = BasicTransformerBlock(ch, context_dim, n_heads)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x))
        h = h.view(B, C, H * W).transpose(1, 2)
        h = self.block(h, context)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return residual + self.proj_out(h)



class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, context_dim, n_heads, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.attn = SpatialTransformer(out_ch, context_dim, n_heads) if use_attn else None

    def forward(self, x, t_emb, context):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, context_dim, n_heads, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.attn = SpatialTransformer(out_ch, context_dim, n_heads) if use_attn else None

    def forward(self, x, skip, t_emb, context):
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        return x


class LatentDiffusionModel(nn.Module):


    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 4),
        time_dim: int = 512,
        context_dim: int = 512,
        num_heads: int = 8,
        text_seq_len: int = 77,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gc = gradient_checkpointing

        ch   = [base_channels * m for m in channel_multipliers]  
        nh   = num_heads
        ctx  = context_dim
        t_dim = time_dim

        
        self.time_emb = TimestepEmbedding(base_channels, t_dim)

        
        
        self.null_context = nn.Parameter(torch.zeros(1, text_seq_len, context_dim))

        
        self.init_conv = nn.Conv2d(latent_channels, ch[0], 3, padding=1)

        
        use_attn = [True] * len(ch)   


        
        self.down_blocks  = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = ch[0]
        for i in range(len(ch)):
            out_ch = ch[i]
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, t_dim, ctx, nh, use_attn=use_attn[i])
            )
            if i < len(ch) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.downsamplers.append(None)
            in_ch = out_ch

        
        self.mid_res1 = ResBlock(ch[-1], ch[-1], t_dim)
        self.mid_attn = SpatialTransformer(ch[-1], ctx, nh)
        self.mid_res2 = ResBlock(ch[-1], ch[-1], t_dim)

        
        
        
        
        
        self.up_blocks  = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        ch_rev = list(reversed(ch))
        in_ch  = ch_rev[0]
        for i in range(len(ch_rev)):
            skip_ch = ch_rev[i]
            out_ch  = ch_rev[i]
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, t_dim, ctx, nh,
                        use_attn=use_attn[len(ch) - 1 - i])
            )
            if i < len(ch_rev) - 1:
                next_ch = ch_rev[i + 1]
                self.upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_ch, next_ch, 3, padding=1),
                ))
                in_ch = next_ch
            else:
                self.upsamplers.append(None)
                in_ch = out_ch

        
        self.norm_out = nn.GroupNorm(min(32, ch[0]), ch[0], eps=1e-6)
        self.act_out  = nn.SiLU()
        self.conv_out = nn.Conv2d(ch[0], latent_channels, 3, padding=1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LatentDiffusionModel] {n_params:,} parameters  |  channels={ch}  |  time_dim={time_dim}")

    

    def expanded_null_context(self, batch_size: int) -> torch.Tensor:

        return self.null_context.expand(batch_size, -1, -1)

    def _run_block(self, use_gc, block, *args):

        if use_gc and self.training:
            return checkpoint_utils.checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:

        t_emb = self.time_emb(t)

        
        h = self.init_conv(zt)
        skips = []
        for i, (down_block, downsampler) in enumerate(zip(self.down_blocks, self.downsamplers)):
            h = self._run_block(self.use_gc, down_block, h, t_emb, context)
            skips.append(h)
            if downsampler is not None:
                h = downsampler(h)

        
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        
        for i, (up_block, upsampler) in enumerate(zip(self.up_blocks, self.upsamplers)):
            skip = skips[-(i + 1)]
            h = self._run_block(self.use_gc, up_block, h, skip, t_emb, context)
            if upsampler is not None:
                h = upsampler(h)

        
        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)





ConditionalUNet = LatentDiffusionModel
