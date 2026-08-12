# Shared Vision Transformer backbone for CLIP and DINO.
# Mostly reffered from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=16,embed_dim=384):
        super().__init__()
        assert img_size%patch_size==0,("img_size must be divisible by patch_size!")
        self.img_size=img_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.num_patches=(img_size//patch_size)**2  # 196
        self.proj=nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size,bias=True)# Conv2d replaces explicit unfold + linear projection

    def forward(self,x):
        B,C,H,W=x.shape
        assert H==self.img_size and W==self.img_size,(f"Not a {self.img_size}x{self.img_size} image!")
        x=self.proj(x) # (B,3,224,224) -> Conv2d -> (B,embed_dim,H/patch,W/patch)=(B,384,14,14)
        x=x.flatten(2).transpose(1,2) #(B,384,14,14) -> flatten(2) -> (B,384,196) -> (B,196,384)
        return x # (B,N,embed_dim)  N=196,embed_dim=384
    
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_dim,dropout=0.0):
        super().__init__()
        # Pre-LayerNorms
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention
        self.attn=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        # MLP:Linear -> GELU -> Linear Dimensions:embed_dim -> mlp_dim -> embed_dim (384 -> 1536 -> 384)
        self.mlp=nn.Sequential(nn.Linear(embed_dim,mlp_dim),nn.GELU(),nn.Linear(mlp_dim,embed_dim))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None,):
        # Self-Attention sub-layer (Pre-LN)
        normed=self.norm1(x)
        attn_out,_=self.attn(query=normed,key=normed,value=normed,attn_mask=mask,need_weights=False)
        x=x + self.dropout(attn_out)
        # MLP sub-layer (Pre-LN)
        x=x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,img_size=224,patch_size=16,embed_dim=384,depth=12,num_heads=6,mlp_dim=1536,dropout:float=0.0):
        super().__init__()
        self.patch_size=patch_size
        self.embed_dim=embed_dim

        # 1. Patch embedding
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches # 196 for 224×224

        # 2. [CLS] token i.e. a Learnable vector prepended to the patch sequence.
        self.cls_token=nn.Parameter(torch.zeros(1, 1, embed_dim))# Shape (1, 1, embed_dim) — broadcast over batch dimension.

        # 3. Positional embeddings, where Length=1 (CLS) + num_patches (196)=197.
        self.pos_embed=nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop=nn.Dropout(dropout)

        # 4. Transformer blocks
        self.blocks=nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim,num_heads=num_heads,mlp_dim=mlp_dim,dropout=dropout)
            for _ in range(depth)
        ])

        # 5. Final LayerNorm
        self.norm=nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _interpolate_pos_embed(self, x, H, W):
        N_native=self.pos_embed.shape[1] - 1   # 196
        grid_native=int(N_native ** 0.5)# 14
        if H==grid_native and W==grid_native:# no interpolation needed
            return self.pos_embed                 
        cls_pos=self.pos_embed[:, :1, :] # (1, 1, embed_dim)
        patch_pos=self.pos_embed[:, 1:, :] # (1, 196, embed_dim)
        patch_pos=patch_pos.reshape(1, grid_native, grid_native, self.embed_dim).permute(0, 3, 1, 2)  # Reshape to spatial grid for interpolation i.e from (1, 196, embed_dim) to (1, embed_dim, 14, 14)
        patch_pos=torch.nn.functional.interpolate(patch_pos,size=(H, W),mode="bicubic",align_corners=False)
        patch_pos=patch_pos.flatten(2).transpose(1, 2)# (1, embed_dim, H, W) -> (1, H*W, embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)  # (1, H*W+1, embed_dim)


    def forward(self, x):
        B, C, H, W=x.shape

        # current patch grid dimensions
        h_grid=H // self.patch_size   # 14 for 224, 6 for 96
        w_grid=W // self.patch_size

        # Patch embedding
        x=self.patch_embed.proj(x) # (B, embed_dim, h_grid, w_grid)
        x=x.flatten(2).transpose(1, 2) # (B, N, embed_dim)

        # Prepend [CLS] token
        cls_tokens=self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x=torch.cat([cls_tokens, x], dim=1) # (B, N+1, embed_dim)

        # Add positional embeddings
        pos_emb=self._interpolate_pos_embed(x, h_grid, w_grid) # effectively it is interpolation if needed
        x=self.pos_drop(x + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x=block(x, mask=None) # no causal mask needed for vision related task

        x=self.norm(x) # (B, N+1, embed_dim)

        # Split CLS and patch tokens
        cls_token=x[:, 0] # (B, embed_dim)
        patch_embs=x[:, 1:] # (B, N, embed_dim)

        return cls_token, patch_embs

# # Smoke-test
# if __name__=="__main__":

#     # ── PatchEmbed smoke-test (from Step 4) ─────────────────────────────
#     patch_embed=PatchEmbed(img_size=224,patch_size=16,embed_dim=384)
#     dummy_img=torch.zeros(2,3,224,224)
#     patches=patch_embed(dummy_img)
#     assert patches.shape==(2,196,384)
#     print(f"PatchEmbed  :{patches.shape}  (Passed)")

#     # ── TransformerBlock — vision mode (no mask) ─────────────────────────
#     block=TransformerBlock(embed_dim=384,num_heads=6,mlp_dim=1536)
#     out=block(patches)                          # mask=None
#     assert out.shape==(2,196,384),f"Unexpected shape:{out.shape}"
#     print(f"TransformerBlock (vision,no mask) :{out.shape}  (Passed)")

#     # ── TransformerBlock — text mode (causal mask) ───────────────────────
#     seq_len=40
#     dummy_seq=torch.zeros(2,seq_len,384)

#     # Upper-triangular causal mask:positions j > i get -inf so token i
#     # cannot attend to future token j.
#     causal_mask=torch.triu(
#         torch.full((seq_len,seq_len),float("-inf")),
#         diagonal=1,
#     )
#     out_text=block(dummy_seq,mask=causal_mask)
#     assert out_text.shape==(2,seq_len,384)
#     print(f"TransformerBlock (text,causal mask):{out_text.shape}  (Passed)")

#     # ── Parameter count ──────────────────────────────────────────────────
#     n_params=sum(p.numel() for p in block.parameters())
#     print(f"TransformerBlock param count:{n_params:,}")

#     print("TransformerBlock smoke-test passed.")

#     vit=VisionTransformer(
#         img_size=224,
#         patch_size=16,
#         embed_dim=384,
#         depth=12,
#         num_heads=6,
#         mlp_dim=1536,
#     )

#     total_params=sum(p.numel() for p in vit.parameters())
#     print(f"ViT-S/16 total parameters :{total_params:,}")
#     # Expected ≈ 21-22M for ViT-Small

#     # ── Global crop (224×224) ────────────────────────────────────────────
#     img_global=torch.zeros(2, 3, 224, 224)
#     cls, patches=vit(img_global)
#     assert cls.shape==(2, 384),       f"CLS shape wrong:{cls.shape}"
#     assert patches.shape==(2, 196, 384),  f"Patch shape wrong:{patches.shape}"
#     print(f"Global crop  224×224 -> cls {cls.shape}, patches {patches.shape}  (Passed)")

#     # ── Local crop (96×96) — DINO only ──────────────────────────────────
#     img_local=torch.zeros(2, 3, 96, 96)
#     cls_l, patches_l=vit(img_local)
#     assert cls_l.shape==(2, 384),      f"CLS shape wrong:{cls_l.shape}"
#     assert patches_l.shape==(2, 36, 384),  f"Patch shape wrong:{patches_l.shape}"
#     print(f"Local  crop   96×96  -> cls {cls_l.shape}, patches {patches_l.shape}  (Passed)")

#     # ── GAP (Global Average Pool over patches) ───────────────────────────
#     gap=patches.mean(dim=1)
#     assert gap.shape==(2, 384)
#     print(f"GAP embedding shape  :{gap.shape}  (Passed)")

#     print("VisionTransformer smoke-test passed.")