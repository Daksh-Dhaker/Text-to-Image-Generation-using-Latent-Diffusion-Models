"""
Usage:
    python -m part_b.train_stage2 \
        --data-root /path/to/Part_A \
        --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
        --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
        --output-dir ./outputs/vlm_stage2 \
        --epochs 3 --batch-size 2 --accum-steps 8

Directory layout expected under --data-root:
    data_root/
      Clevr_official/images/{train,val}/
      Probe-Datasets/CLEVR_{train,val}_explanations_v0.7.10.json
      Probe-Datasets/{train|dev}_images_ids_v0.7.10-recut.pkl
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW = False

from models.vit import VisionTransformer 

from dataset_vlm import QADataset, vlm_collate_fn
from models.projector import ReverseBNProjector
from models.vlm import VLMModel

from transformers import AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError as e:
    raise ImportError("peft is required for Stage 2. `pip install peft`.") from e


def build_vlm_image_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def apply_lora_to_qwen(qwen_model, r=16, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                       lora_alpha=32, lora_dropout=0.05):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(qwen_model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ── Model construction ────────────────────────────────────────────────────────

def build_stage2_model(vit_ckpt_path, qwen_name, stage1_ckpt, lora_r, lora_alpha, lora_dropout,
                        image_token_id, device, gradient_checkpointing=True):
    projector = ReverseBNProjector(in_dim=384, hidden_dim=1536, out_dim=2560)
    model = VLMModel(
        vit_checkpoint_path=vit_ckpt_path,
        projector=projector,
        qwen_model_name=qwen_name,
        image_token_id=image_token_id,
    )

    s1 = torch.load(stage1_ckpt, map_location="cpu")
    projector_sd = s1.get("projector", s1)
    model.projector.load_state_dict(projector_sd)
    print(f"[build_stage2_model] Loaded Stage-1 projector from {stage1_ckpt}")

    for p in model.vit.parameters():
        p.requires_grad = False
    model.vit.eval()

    model.qwen = apply_lora_to_qwen(
        model.qwen,
        r=lora_r,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    for p in model.projector.parameters():
        p.requires_grad = True

    if gradient_checkpointing:
        base = getattr(model.qwen, "base_model", None)
        inner = getattr(base, "model", model.qwen) if base is not None else model.qwen
        if hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable()
            if hasattr(inner, "enable_input_require_grads"):
                inner.enable_input_require_grads()
        elif hasattr(model.qwen, "gradient_checkpointing_enable"):
            model.qwen.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[build_stage2_model] trainable={n_trainable/1e6:.2f}M / "
        f"total={n_total/1e6:.2f}M ({100*n_trainable/n_total:.3f}%)"
    )
    return model.to(device)


def build_param_groups(model, projector_lr, lora_lr, weight_decay):
    projector_params, lora_params, other_trainable = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("projector."):
            projector_params.append(p)
        elif "lora_" in name.lower():
            lora_params.append(p)
        else:
            other_trainable.append(p)

    groups = [
        {"params": projector_params, "lr": projector_lr, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
    ]
    if other_trainable:
        groups.append({"params": other_trainable, "lr": lora_lr, "weight_decay": weight_decay})
    return groups


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): return False

def train_stage2_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                            device, accum_steps=8):
    grad_clip = 1.0
    use_amp = (device.type == "cuda")

    model.train()
    model.vit.eval()

    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_amp and _AMP_NEW:
            amp_ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
        elif use_amp:
            amp_ctx = autocast()
        else:
            amp_ctx = _NullCtx()

        with amp_ctx:
            out = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * accum_steps
        n_batches += 1
        if step % 100 == 0:
            print(f"  [train] step {step+1}/{len(dataloader)} loss={loss.item()*accum_steps:.4f}", flush=True)

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_stage2(model, dataloader, device):
    use_amp = (device.type == "cuda")
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_amp and _AMP_NEW:
            amp_ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
        elif use_amp:
            amp_ctx = autocast()
        else:
            amp_ctx = _NullCtx()

        with amp_ctx:
            out = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_stage2_checkpoint(out_dir, model, optimizer, scheduler, scaler, epoch, best_val_loss, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model.qwen, PeftModel):
        model.qwen.save_pretrained(str(tag_dir / "lora"))
    else:
        lora_sd = {k: v for k, v in model.qwen.state_dict().items() if "lora_" in k.lower()}
        torch.save(lora_sd, tag_dir / "lora_adapter.pt")

    torch.save(
        {
            "epoch": epoch,
            "projector": model.projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
        },
        tag_dir / "state.pt",
    )


def load_stage2_checkpoint(tag_dir, model, optimizer=None, scheduler=None, scaler=None):
    state_path = tag_dir / "state.pt"
    ckpt = torch.load(state_path, map_location="cpu")
    model.projector.load_state_dict(ckpt["projector"])

    lora_dir = tag_dir / "lora"
    if lora_dir.is_dir() and isinstance(model.qwen, PeftModel):
        model.qwen.load_adapter(str(lora_dir), adapter_name="default", is_trainable=True)
    elif (tag_dir / "lora_adapter.pt").exists():
        lora_sd = torch.load(tag_dir / "lora_adapter.pt", map_location="cpu")
        model.qwen.load_state_dict(lora_sd, strict=False)

    start_epoch = ckpt.get("epoch", -1) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return start_epoch, best_val_loss


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 VLM training (LoRA + projector)")
    p.add_argument("--data-root", type=str, required=True,help="Root of Part-A data (contains Clevr_official/ and Probe-Datasets/).")
    p.add_argument("--train-images", type=str, default=None)
    p.add_argument("--val-images", type=str, default=None)

    p.add_argument("--vit-ckpt", type=str, required=True)
    p.add_argument("--vit-kind", type=str, default="raw",
                   choices=["clip_vit", "dino_student", "dino_teacher", "raw"])
    p.add_argument("--qwen-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--stage1-ckpt", type=str, required=True)

    p.add_argument("--output-dir", type=str, default="./outputs/vlm_stage2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap QA samples per split (e.g. 50000) to limit epoch time")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--projector-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)

    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_vit_snapshot(args, output_dir):
    if args.vit_kind == "raw":
        return Path(args.vit_ckpt)
    from train_stage1 import save_vit_snapshot
    clean_path = output_dir / "vit_clean.pt"
    if not clean_path.exists():
        save_vit_snapshot(args.vit_ckpt, args.vit_kind, clean_path)
    return clean_path


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_clean_path = _prepare_vit_snapshot(args, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_name, trust_remote_code=True)#NEED TO BE FIXED FETCHES FROM WEB, NEEDS INTERNET
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>"]}
    )
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[main] <image> token id = {image_token_id}")

    transform = build_vlm_image_transform(224)

    train_ds = QADataset(
        root=args.data_root, split="train",
        transform=transform, training=True,
        max_samples=args.max_samples,
    )
    val_ds = QADataset(
        root=args.data_root, split="val",
        transform=transform, training=False,
        max_samples=args.max_samples,
    )
    print(f"[main] train_ds={len(train_ds)} samples, val_ds={len(val_ds)} samples")

    collate_fn = vlm_collate_fn(tokenizer, max_len=args.max_len, stage=2)# vlm_collate_fn is a factory; stage=2 handles raw QA tuples internally.

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )

    model = build_stage2_model(
        vit_ckpt_path=str(vit_clean_path),
        qwen_name=args.qwen_name,
        stage1_ckpt=args.stage1_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_token_id=image_token_id,
        device=device,
        gradient_checkpointing=not args.no_grad_ckpt,
    )

    param_groups = build_param_groups(
        model,
        projector_lr=args.projector_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = AdamW(param_groups, betas=(args.beta1, args.beta2), eps=args.eps)

    effective_steps_per_epoch = max(len(train_loader) // args.accum_steps, 1)
    total_opt_steps = args.epochs * effective_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_opt_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_opt_steps)

    use_amp = (not args.no_amp) and device.type == "cuda"
    if use_amp and _AMP_NEW:
        scaler = GradScaler(device="cuda", enabled=False)
    elif use_amp:
        scaler = GradScaler(enabled=False)
    else:
        scaler = None

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_stage2_checkpoint(Path(args.resume), model, optimizer, scheduler, scaler)

    metrics_csv = metrics_dir / "train_log.csv"
    if not metrics_csv.exists() or start_epoch == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr_projector", "lr_lora"])

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_stage2_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            accum_steps=args.accum_steps,
        )

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate_stage2(model, val_loader, device)

        lr_proj = optimizer.param_groups[0]["lr"]
        lr_lora = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_proj

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr_proj, lr_lora])

        save_stage2_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, epoch, best_val_loss, tag="last")

        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_stage2_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler, epoch, best_val_loss, tag="best")

        if (epoch + 1) % args.save_every == 0:
            save_stage2_checkpoint(
                ckpt_dir, model, optimizer, scheduler, scaler, epoch, best_val_loss,
                tag=f"epoch_{epoch + 1:03d}",
            )

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": (round(val_loss, 6) if not math.isnan(val_loss) else None),
            "lr_projector": lr_proj,
            "lr_lora": lr_lora,
            "best_val_loss": (round(best_val_loss, 6) if best_val_loss != float("inf") else None),
        }))


if __name__ == "__main__":
    main()