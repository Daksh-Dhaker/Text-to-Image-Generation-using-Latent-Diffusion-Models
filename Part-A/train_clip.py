import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import CLEVRCaptionDataset, SimpleTokenizer
from models.clip import CLIPModel, TextTransformer, clip_loss
from models.vit import VisionTransformer


_COLLATE_TOKENIZER = None
_COLLATE_MAX_LEN = 40


def configure_clip_collate(tokenizer: SimpleTokenizer, max_len: int) -> None:
    global _COLLATE_TOKENIZER, _COLLATE_MAX_LEN
    _COLLATE_TOKENIZER = tokenizer
    _COLLATE_MAX_LEN = max_len


def build_clip_image_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def clip_collate_fn(batch):
    if _COLLATE_TOKENIZER is None:
        raise RuntimeError("clip_collate_fn is not configured. Call configure_clip_collate first.")

    images = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    token_ids = torch.tensor([_COLLATE_TOKENIZER.encode(c, max_len=_COLLATE_MAX_LEN) for c in captions], dtype=torch.long)

    batch_size = len(captions)
    valid_pair_mask = torch.ones((batch_size, batch_size), dtype=torch.bool)
    caption_to_indices = {}
    for i, cap in enumerate(captions):
        caption_to_indices.setdefault(cap, []).append(i)

    for idxs in caption_to_indices.values():
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    if i != j:
                        valid_pair_mask[i, j] = False

    return images, token_ids, valid_pair_mask


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_clip(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, token_ids, valid_pair_mask in dataloader:
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        valid_pair_mask = valid_pair_mask.to(device, non_blocking=True)

        image_emb, text_emb, logit_scale = model(images, token_ids)
        loss = clip_loss(image_emb, text_emb, logit_scale, valid_pair_mask=valid_pair_mask)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_clip_one_epoch(model, dataloader, optimizer, scheduler, device, grad_clip: float = 1.0):
    model.train()
    running_loss = 0.0

    for images, token_ids, valid_pair_mask in dataloader:
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        valid_pair_mask = valid_pair_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        image_emb, text_emb, logit_scale = model(images, token_ids)
        loss = clip_loss(image_emb, text_emb, logit_scale, valid_pair_mask=valid_pair_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return start_epoch, best_val_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP on CLEVR captions")
    parser.add_argument("--data-root", type=str, default="./data/partA")
    parser.add_argument("--output-dir", type=str, default="./outputs/clip")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
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

    train_transform = build_clip_image_transform(224)
    val_transform = build_clip_image_transform(224)

    train_ds = CLEVRCaptionDataset(root=args.data_root, split="train", transform=train_transform)
    val_ds = CLEVRCaptionDataset(root=args.data_root, split="val", transform=val_transform)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_ds.get_all_captions())
    tokenizer.save(str(output_dir / "tokenizer.json"))

    configure_clip_collate(tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=clip_collate_fn,
    )

    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
    )
    txt = TextTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_len,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_dim=1536,
        pad_id=SimpleTokenizer.PAD_ID,
    )
    model = CLIPModel(vision_encoder=vit, text_encoder=txt, embed_dim=512).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(Path(args.resume), model, optimizer, scheduler)

    metrics_csv = metrics_dir / "train_log.csv"
    if not metrics_csv.exists() or start_epoch == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_clip_one_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=args.grad_clip)
        val_loss = evaluate_clip(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        save_checkpoint(ckpt_dir / "last.pt", payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload["best_val_loss"] = best_val_loss
            save_checkpoint(ckpt_dir / "best.pt", payload)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:03d}.pt", payload)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": lr,
                    "best_val_loss": round(best_val_loss, 6),
                }
            )
        )


if __name__ == "__main__":
    main()
