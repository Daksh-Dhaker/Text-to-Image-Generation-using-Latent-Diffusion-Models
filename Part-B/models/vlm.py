import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vit import VisionTransformer 
from .projector import ReverseBNProjector 

_VIT_IMG_SIZE=224
_VIT_PATCH_SIZE=16
_VIT_EMBED_DIM=384
_VIT_DEPTH=12
_VIT_HEADS=6
_VIT_MLP_DIM=1536
_IGNORE_IDX=-100

def merge_visual_tokens(
    text_embeds,
    image_tokens,
    attention_mask,
    labels,
    placeholder_idx,
):
    B, T, D=text_embeds.shape
    _, N_img, _=image_tokens.shape
    p=placeholder_idx
    before=text_embeds[:, :p, :]          
    after=text_embeds[:, p + 1:, :]     
    merged_embeds=torch.cat([before, image_tokens, after], dim=1)
    vis_attn=torch.ones(
        B, N_img, dtype=attention_mask.dtype, device=attention_mask.device
    )
    merged_attn=torch.cat(
        [attention_mask[:, :p], vis_attn, attention_mask[:, p + 1:]], dim=1
    )
    vis_labels=torch.full(
        (B, N_img), fill_value=_IGNORE_IDX,
        dtype=labels.dtype, device=labels.device,
    )
    merged_labels=torch.cat(
        [labels[:, :p], vis_labels, labels[:, p + 1:]], dim=1
    )
    return merged_embeds, merged_attn, merged_labels

def _resolve_embed_fn(qwen):
    """Traverse PeftModel → Qwen3ForCausalLM → Qwen3Model to find embed_tokens.
    Works for both plain Qwen3ForCausalLM (stage-1) and PeftModel-wrapped (stage-2)."""
    m = qwen
    while not hasattr(m, "embed_tokens"):
        if not hasattr(m, "model"):
            raise AttributeError(
                f"Cannot find embed_tokens in {type(m).__name__}. "
                "Expected Qwen3ForCausalLM or PeftModel wrapping it."
            )
        m = m.model
    return m.embed_tokens

class VLMModel(nn.Module):
    def __init__(
        self,
        vit_checkpoint_path:str,
        projector:ReverseBNProjector,
        qwen_model_name:str="Qwen/Qwen3-4B-Instruct-2507",
        image_token_id=None,
    ):
        super().__init__()
        self.vit=VisionTransformer(
            img_size=_VIT_IMG_SIZE,
            patch_size=_VIT_PATCH_SIZE,
            embed_dim=_VIT_EMBED_DIM,
            depth=_VIT_DEPTH,
            num_heads=_VIT_HEADS,
            mlp_dim=_VIT_MLP_DIM,
        )
        self._load_vit_weights(vit_checkpoint_path)
        self._freeze_module(self.vit)
        self.projector=projector
        print(f"Loading Qwen model:{qwen_model_name}")#NEED TO FIX: USES INTERNET
        self.qwen=AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,   
            device_map=None,             
            trust_remote_code=True,
        )

        if image_token_id is not None:
            vocab_size_with_image=image_token_id + 1
            current_vocab=self.qwen.config.vocab_size
            if vocab_size_with_image > current_vocab:
                self.qwen.resize_token_embeddings(vocab_size_with_image)
                print(
                    f"  Resized Qwen embedding table:"
                    f"{current_vocab} → {vocab_size_with_image}"
                )

        self.image_token_id=image_token_id
        self._qwen_hidden=self.qwen.config.hidden_size

        assert self.projector.out_dim==self._qwen_hidden, (
            f"Projector out_dim ({self.projector.out_dim}) does not match "
            f"Qwen hidden_size ({self._qwen_hidden}).  "
            f"Reconstruct projector with out_dim={self._qwen_hidden}."
        )

    def _load_vit_weights(self, ckpt_path:str):
        ckpt_path=Path(ckpt_path)
        assert ckpt_path.is_file(), f"ViT checkpoint not found:{ckpt_path}"
        raw=torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            state=raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            state=raw["model"]
        elif isinstance(raw, dict):
            state=raw
        else:
            raise ValueError(
                f"Unrecognised checkpoint format in {ckpt_path}. "
                "Expected a dict with keys 'state_dict', 'model', or raw."
            )
        
        _VIT_PREFIXES=[
            "",                  
            "vit.",              
            "vision_encoder.",   
            "image_encoder.",    
            "student.vit.",      
            "teacher.vit.",      
        ]

        vit_keys=set(self.vit.state_dict().keys())

        for prefix in _VIT_PREFIXES:
            if prefix=="":
                candidate={k:v for k, v in state.items() if k in vit_keys}
            else:
                candidate={
                    k[len(prefix):]:v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
            
            if "cls_token" in candidate and "patch_embed.proj.weight" in candidate:
                missing, unexpected=self.vit.load_state_dict(
                    candidate, strict=True
                )
                print(
                    f"  ViT weights loaded from '{ckpt_path}' "
                    f"(prefix='{prefix}')"
                )
                return

        raise RuntimeError(
            f"Could not extract ViT weights from checkpoint '{ckpt_path}'. "
            f"Top-level keys:{list(state.keys())[:10]} ..."
        )

    @staticmethod
    def _freeze_module(module:nn.Module):
        for param in module.parameters():
            param.requires_grad=False

    def freeze_for_stage1(self):
        self._freeze_module(self.vit)
        self._freeze_module(self.qwen)
        for param in self.projector.parameters():
            param.requires_grad=True

        trainable=sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total=sum(p.numel() for p in self.parameters())
        print(
            f"Stage-1 trainable params:{trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    @torch.no_grad()
    def _encode_vit(self, images) -> torch.Tensor:
        _, patch_embs=self.vit(images)   
        return patch_embs

    def encode_image(self, images) -> torch.Tensor:
        patch_embs=self._encode_vit(images)         
        visual_tokens=self.projector(patch_embs)       
        return visual_tokens

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        labels=None,
    ) -> CausalLMOutputWithPast:

        assert self.image_token_id is not None, (
            "image_token_id not set.  Pass image_token_id to VLMModel.__init__."
        )
        visual_tokens=self.encode_image(images)   
        visual_tokens=visual_tokens.to(
            dtype=next(self.qwen.parameters()).dtype
        )
        embed_fn=_resolve_embed_fn(self.qwen)
        text_embeds=embed_fn(input_ids)           
        is_placeholder=(input_ids==self.image_token_id)  
        assert is_placeholder.any(), (
            "No <image> token found in input_ids.  "
            "Ensure the prompt includes the <image> placeholder and the "
            "tokenizer has it registered as a special token."
        )
        placeholder_idx=int(is_placeholder[0].nonzero(as_tuple=False)[0, 0])
        if labels is None:
            _dummy_labels=torch.full_like(input_ids, _IGNORE_IDX)
        else:
            _dummy_labels=labels

        merged_embeds, merged_attn, merged_labels=merge_visual_tokens(
            text_embeds=text_embeds,
            image_tokens=visual_tokens,
            attention_mask=attention_mask,
            labels=_dummy_labels,
            placeholder_idx=placeholder_idx,
        )
        output=self.qwen(
            inputs_embeds=merged_embeds,
            attention_mask=merged_attn,
            labels=merged_labels if labels is not None else None,
            return_dict=True,
        )
        return output

    def print_trainable_parameters(self):
        trainable=0
        frozen=0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable +=param.numel()
            else:
                frozen +=param.numel()
        total=trainable + frozen
        print(
            f"Trainable :{trainable:>15,}  ({100*trainable/total:.2f}%)\n"
            f"Frozen    :{frozen:>15,}  ({100*frozen/total:.2f}%)\n"
            f"Total     :{total:>15,}"
        )