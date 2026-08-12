import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import CLEVRImageDataset, DINOAugment
from models.dino import DINOLoss, DINOHead, DINOModel, update_teacher_ema
from models.vit import VisionTransformer


_DINO_EPOCH = 0
_DINO_TOTAL_EPOCHS = 1
_DINO_EMA_BASE = 0.996
_DINO_EMA_FINAL = 1.0
_DINO_GRAD_CLIP = 1.0


def configure_dino_epoch(epoch: int, total_epochs: int, ema_base: float, ema_final: float, grad_clip: float) -> None:
    global _DINO_EPOCH, _DINO_TOTAL_EPOCHS, _DINO_EMA_BASE, _DINO_EMA_FINAL, _DINO_GRAD_CLIP
    _DINO_EPOCH = epoch
    _DINO_TOTAL_EPOCHS = max(total_epochs, 1)
    _DINO_EMA_BASE = ema_base
    _DINO_EMA_FINAL = ema_final
    _DINO_GRAD_CLIP = grad_clip


def dino_collate_fn(batch):
    n_crops = len(batch[0])
    out = []
    for i in range(n_crops):
        out.append(torch.stack([sample[i] for sample in batch], dim=0))
    return out


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def momentum_schedule(base_m: float, final_m: float, current_step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return final_m
    ratio = current_step / (total_steps - 1)
    return final_m - (final_m - base_m) * (math.cos(math.pi * ratio) + 1) / 2


def train_dino_one_epoch(
    student,
    teacher,
    loss_fn,
    dataloader,
    optimizer,
    scheduler,
    device,
):
    student.train()
    teacher.eval()

    total_steps_epoch = max(len(dataloader), 1)
    total_steps_run = _DINO_TOTAL_EPOCHS * total_steps_epoch
    running_loss = 0.0

    for step, crops in enumerate(dataloader):
        crops = [c.to(device, non_blocking=True) for c in crops]

        optimizer.zero_grad(set_to_none=True)
        student_out = student(crops)
        with torch.no_grad():
            teacher_out = teacher(crops[:2])

        loss = loss_fn(student_out, teacher_out, n_global_crops=2, n_local_crops=max(len(crops) - 2, 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=_DINO_GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        global_step = _DINO_EPOCH * total_steps_epoch + step
        m = momentum_schedule(_DINO_EMA_BASE, _DINO_EMA_FINAL, global_step, total_steps_run)
        update_teacher_ema(student, teacher, m)

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINO on CLEVR images")
    parser.add_argument("--data-root", type=str, default="./data/partA")
    parser.add_argument("--output-dir", type=str, default="./outputs/dino")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-local-crops", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--ema-base", type=float, default=0.996)
    parser.add_argument("--ema-final", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
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

    train_ds = CLEVRImageDataset(root=args.data_root, split="train", transform=DINOAugment(n_local_crops=args.n_local_crops))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=dino_collate_fn,
    )

    backbone_student = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
    )
    student = DINOModel(backbone=backbone_student, head=DINOHead(in_dim=384, out_dim=4096)).to(device)

    # teacher = deepcopy(student).to(device)
    backbone_teacher = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
    )
    teacher = DINOModel(backbone=backbone_teacher, head=DINOHead(in_dim=384, out_dim=4096)).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = DINOLoss(
        out_dim=4096,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    optimizer = AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    metrics_csv = metrics_dir / "train_log.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "lr"])

    for epoch in range(args.epochs):
        configure_dino_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            ema_base=args.ema_base,
            ema_final=args.ema_final,
            grad_clip=args.grad_clip,
        )
        train_loss = train_dino_one_epoch(
            student,
            teacher,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            device,
        )
        lr = optimizer.param_groups[0]["lr"]

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, lr])

        payload = {
            "epoch": epoch,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "loss_center": loss_fn.center,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }

        save_checkpoint(ckpt_dir / "last.pt", payload)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:03d}.pt", payload)

        print(json.dumps({"epoch": epoch, "train_loss": round(train_loss, 6), "lr": lr}))


if __name__ == "__main__":
    main()
