"""
Usage:
    python -m part_b.train_stage1 \
        --data-root /path/to/Part_A \
        --vit-ckpt /path/to/clip_or_dino_ckpt.pt \
        --vit-kind clip_vit \
        --output-dir ./outputs/vlm_stage1 \
        --epochs 5 --batch-size 16 --lr 1e-3

Directory layout expected under --data-root:
    data_root/
      Clevr_official/images/{train,val}/
      Probe-Datasets/clevr_{train,val}_captions.json
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

from models.vit import VisionTransformer

from dataset_vlm import CaptioningDataset, vlm_collate_fn
from models.projector import ReverseBNProjector
from models.vlm import VLMModel

from transformers import AutoTokenizer


def build_vlm_image_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_vit_state_dict(vit_ckpt_path, vit_kind="raw"):
    ckpt = torch.load(vit_ckpt_path, map_location="cpu")
    if vit_kind == "raw":
        return ckpt
    if vit_kind == "clip_vit":
        full_sd, prefix = ckpt["model"], "vision_encoder."
    elif vit_kind == "dino_student":
        full_sd, prefix = ckpt["student"], "backbone."
    elif vit_kind == "dino_teacher":
        full_sd, prefix = ckpt["teacher"], "backbone."
    else:
        raise ValueError(f"Unknown vit_kind: {vit_kind}")
    vit_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
    if not vit_sd:
        raise RuntimeError(f"No keys with prefix '{prefix}' found. Wrong --vit-kind?")
    return vit_sd


def save_vit_snapshot(vit_ckpt_path, vit_kind, out_path):
    vit_sd = load_vit_state_dict(vit_ckpt_path, vit_kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vit_sd, out_path)
    return out_path


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_stage1_model(vit_ckpt_path, qwen_name, image_token_id):
    projector = ReverseBNProjector(in_dim=384, hidden_dim=1536, out_dim=2560)
    model = VLMModel(
        vit_checkpoint_path=vit_ckpt_path,
        projector=projector,
        qwen_model_name=qwen_name,
        image_token_id=image_token_id,
    )
    model.freeze_for_stage1()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[build_stage1_model] trainable={n_trainable/1e6:.2f}M / "
        f"total={n_total/1e6:.2f}M ({100*n_trainable/n_total:.3f}%)"
    )
    return model


def train_stage1_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.projector.train()
    model.vit.eval()
    model.qwen.eval()

    grad_clip = 1.0
    running_loss, n_batches = 0.0, 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_stage1(model, dataloader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_stage1_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, extras):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "projector": model.projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "extras": extras,
    }
    torch.save(payload, path)


def load_stage1_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.projector.load_state_dict(ckpt["projector"])
    start_epoch = ckpt.get("epoch", -1) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch, best_val_loss


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 VLM training (projector-only)")
    p.add_argument("--data-root", type=str, required=True)

    p.add_argument("--vit-ckpt", type=str, required=True)
    p.add_argument("--vit-kind", type=str, default="clip_vit",
                   choices=["clip_vit", "dino_student", "dino_teacher", "raw"])
    p.add_argument("--qwen-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")

    p.add_argument("--output-dir", type=str, default="./outputs/vlm_stage1")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-len", type=int, default=256)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)

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


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_clean_path = output_dir / "vit_clean.pt"
    if not vit_clean_path.exists():
        save_vit_snapshot(args.vit_ckpt, args.vit_kind, vit_clean_path)
        print(f"[main] Saved bare ViT state_dict -> {vit_clean_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>"]}
    )
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[main] <image> token id = {image_token_id}")

    transform = build_vlm_image_transform(224)
    
    train_ds = CaptioningDataset(
        root=args.data_root,
        split="train",
        transform=transform,
    )
    val_ds = CaptioningDataset(
        root=args.data_root,
        split="val",
        transform=transform,
    )

    collate_fn = vlm_collate_fn(tokenizer, max_len=args.max_len, stage=1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )

    model = build_stage1_model(
        vit_ckpt_path=str(vit_clean_path),
        qwen_name=args.qwen_name,
        image_token_id=image_token_id,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=args.eps,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_stage1_checkpoint(Path(args.resume), model, optimizer, scheduler)

    metrics_csv = metrics_dir / "train_log.csv"
    if not metrics_csv.exists() or start_epoch == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_stage1_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_stage1(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])

        extras = {"args": vars(args)}
        save_stage1_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss, extras)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_stage1_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss, extras)

        if (epoch + 1) % args.save_every == 0:
            save_stage1_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                model, optimizer, scheduler, epoch, best_val_loss, extras,
            )

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "best_val_loss": round(best_val_loss, 6),
        }))


if __name__ == "__main__":
    main()