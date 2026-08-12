import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CLEVRProbeDataset, SimpleTokenizer
from models.clip import CLIPModel, TextTransformer
from models.dino import DINOHead, DINOModel
from models.vit import VisionTransformer


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return self.fc(x)


def build_probe_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    n_classes = y_true.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = np.logical_and(y_true[:, c] == 1, y_pred[:, c] == 1).sum()
        fp = np.logical_and(y_true[:, c] == 0, y_pred[:, c] == 1).sum()
        fn = np.logical_and(y_true[:, c] == 1, y_pred[:, c] == 0).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


@torch.no_grad()
def extract_features(encoder, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    cls_list, gap_list, labels = [], [], []

    for images, y in dataloader:
        images = images.to(device, non_blocking=True)

        cls, patches = encoder(images)
        gap = patches.mean(dim=1)

        cls_list.append(cls.cpu().numpy())
        gap_list.append(gap.cpu().numpy())
        if torch.is_tensor(y):
            labels.append(y.detach().cpu().numpy())
        else:
            labels.append(np.asarray(y))

    return (
        np.concatenate(cls_list, axis=0),
        np.concatenate(gap_list, axis=0),
        np.concatenate(labels, axis=0),
    )


def train_probe(probe, train_x, train_y, task: str, epochs: int, lr: float):
    device = next(probe.parameters()).device
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    x = torch.tensor(train_x, dtype=torch.float32, device=device)

    if task in {"count", "multiclass"}:
        y = torch.tensor(train_y, dtype=torch.long, device=device)
        criterion = nn.CrossEntropyLoss()
    elif task in {"colors", "multilabel"}:
        y = torch.tensor(train_y, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task: {task}")

    for _ in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    return probe


@torch.no_grad()
def eval_probe(probe, x, y, task: str):
    device = next(probe.parameters()).device
    probe.eval()
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    logits = probe(x_t).cpu().numpy()

    if task in {"count", "multiclass"}:
        preds = np.argmax(logits, axis=1)
        acc = (preds == y).mean().item()
        return acc

    if task not in {"colors", "multilabel"}:
        raise ValueError(f"Unknown task: {task}")

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)
    f1 = compute_macro_f1(y.astype(np.int32), preds)
    return f1


def build_backbone(kind: str, ckpt_path: str, tokenizer_path: str = None, max_len: int = 40):
    if kind == "clip":
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required when --model-kind clip")

        tok = SimpleTokenizer.load(tokenizer_path)
        vocab_size = tok.vocab_size

        vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        txt = TextTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_dim=1536,
            pad_id=SimpleTokenizer.PAD_ID,
        )
        clip_model = CLIPModel(vision_encoder=vit, text_encoder=txt, embed_dim=512)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        clip_model.load_state_dict(ckpt["model"])
        return clip_model.vision_encoder

    if kind in {"dino_student", "dino_teacher"}:
        backbone = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        model = DINOModel(backbone=backbone, head=DINOHead(in_dim=384, out_dim=4096))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = "student" if kind == "dino_student" else "teacher"
        model.load_state_dict(ckpt[key])
        return model.backbone

    raise ValueError(f"Unknown model kind: {kind}")


def parse_args():
    p = argparse.ArgumentParser(description="Linear probing for CLIP and DINO features")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--probe-task", type=str, choices=["count", "colors"], required=True)
    p.add_argument("--model-kind", type=str, choices=["clip", "dino_student", "dino_teacher"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./outputs/probe")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--max-len", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    metrics_dir = out_dir / "metrics"
    feat_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_probe_transform(224)

    train_ds = CLEVRProbeDataset(root=args.data_root, task=args.probe_task, split="train", transform=transform)
    val_ds = CLEVRProbeDataset(root=args.data_root, task=args.probe_task, split="val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    backbone = build_backbone(args.model_kind, args.checkpoint, tokenizer_path=args.tokenizer, max_len=args.max_len).to(device)

    train_cls, train_gap, train_y = extract_features(backbone, train_loader, device)
    val_cls, val_gap, val_y = extract_features(backbone, val_loader, device)

    np.savez_compressed(feat_dir / f"{args.model_kind}_{args.probe_task}_train.npz", cls=train_cls, gap=train_gap, y=train_y)
    np.savez_compressed(feat_dir / f"{args.model_kind}_{args.probe_task}_val.npz", cls=val_cls, gap=val_gap, y=val_y)

    num_classes = train_ds.num_classes if args.num_classes is None else args.num_classes
    metric_name = "accuracy" if args.probe_task == "count" else "f1_macro"

    results: Dict[str, Dict] = {}

    for emb_name, train_x, val_x in [("cls", train_cls, val_cls), ("gap", train_gap, val_gap)]:
        probe = LinearProbe(input_dim=train_x.shape[1], num_classes=num_classes)
        probe = probe.to(device)
        probe = train_probe(probe, train_x, train_y, task=args.probe_task, epochs=args.epochs, lr=args.lr)
        train_score = eval_probe(probe, train_x, train_y, task=args.probe_task)
        val_score = eval_probe(probe, val_x, val_y, task=args.probe_task)
        results[emb_name] = {
            "train": {metric_name: train_score},
            "val": {metric_name: val_score},
        }

        torch.save(probe.state_dict(), out_dir / f"probe_{args.model_kind}_{args.probe_task}_{emb_name}.pt")

    with open(metrics_dir / f"{args.model_kind}_{args.probe_task}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table_path = metrics_dir / f"{args.model_kind}_{args.probe_task}.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["embedding", "split", "metric", "value"])
        for emb_name, split_dict in results.items():
            for split, metric_map in split_dict.items():
                for metric_name, value in metric_map.items():
                    writer.writerow([emb_name, split, metric_name, value])

    print(json.dumps({"saved": str(table_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
