import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CLEVRCaptionDataset_Aa, SimpleTokenizer
from models.clip import CLIPModel, TextTransformer
from models.vit import VisionTransformer


_RETRIEVAL_MAX_LEN = 40
_RETRIEVAL_BATCH_SIZE = 256
_RETRIEVAL_WORKERS = 8
_RETRIEVAL_NUM_EXAMPLES = 10
_RETRIEVAL_OUTPUT_DIR = Path("./outputs/retrieval")


def configure_retrieval(max_len: int, batch_size: int, workers: int, num_examples: int, output_dir: str) -> None:
    global _RETRIEVAL_MAX_LEN, _RETRIEVAL_BATCH_SIZE, _RETRIEVAL_WORKERS, _RETRIEVAL_NUM_EXAMPLES, _RETRIEVAL_OUTPUT_DIR
    _RETRIEVAL_MAX_LEN = max_len
    _RETRIEVAL_BATCH_SIZE = batch_size
    _RETRIEVAL_WORKERS = workers
    _RETRIEVAL_NUM_EXAMPLES = num_examples
    _RETRIEVAL_OUTPUT_DIR = Path(output_dir)


def compute_recall_at_k(query_embs: np.ndarray, gallery_embs: np.ndarray, ground_truth_indices: np.ndarray, k: int) -> float:
    similarity = query_embs @ gallery_embs.T
    topk = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    hits = (topk == ground_truth_indices[:, None]).any(axis=1)
    return float(hits.mean())


@torch.no_grad()
def encode_all_images(model, dataloader, device):
    embs = []
    captions = []
    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        img_emb = model.encode_image(images)
        embs.append(img_emb.cpu().numpy())
        captions.extend(batch["captions"])
    return np.concatenate(embs, axis=0), captions


@torch.no_grad()
def encode_all_texts(model, captions, tokenizer, max_len, batch_size, device):
    embs = []
    for start in range(0, len(captions), batch_size):
        cap_batch = captions[start : start + batch_size]
        token_ids = torch.tensor([tokenizer.encode(c, max_len=max_len) for c in cap_batch], dtype=torch.long, device=device)
        txt_emb = model.encode_text(token_ids)
        embs.append(txt_emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description="CLIP cross-modal retrieval")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="./outputs/retrieval")
    p.add_argument("--num-examples", type=int, default=10)
    return p.parse_args()


def build_clip_image_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def run_retrieval(clip_model, val_dataset, tokenizer, device):
    out_dir = _RETRIEVAL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]
        return {"images": images, "captions": captions}

    loader = DataLoader(
        val_dataset,
        batch_size=_RETRIEVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=_RETRIEVAL_WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )

    image_emb, captions = encode_all_images(clip_model, loader, device)
    text_emb = encode_all_texts(clip_model, captions, tokenizer, _RETRIEVAL_MAX_LEN, _RETRIEVAL_BATCH_SIZE, device)

    gt = np.arange(len(captions), dtype=np.int64)
    sim_i2t = image_emb @ text_emb.T
    sim_t2i = sim_i2t.T

    metrics = {
        "image_to_text_r1": compute_recall_at_k(image_emb, text_emb, gt, k=1),
        "image_to_text_r3": compute_recall_at_k(image_emb, text_emb, gt, k=3),
        "text_to_image_r1": compute_recall_at_k(text_emb, image_emb, gt, k=1),
        "text_to_image_r3": compute_recall_at_k(text_emb, image_emb, gt, k=3),
    }

    idx_map = np.arange(len(captions), dtype=np.int64)
    np.savez_compressed(out_dir / "retrieval_embeddings.npz", image_emb=image_emb, text_emb=text_emb, idx_map=idx_map)

    example_rows = []
    n = min(_RETRIEVAL_NUM_EXAMPLES, len(captions))
    for q in range(n):
        top_t = np.argsort(-sim_i2t[q])[:3].tolist()
        top_i = np.argsort(-sim_t2i[q])[:3].tolist()
        example_rows.append(
            {
                "query_index": int(q),
                "query_caption": captions[q],
                "i2t_top3": [{"idx": int(i), "caption": captions[i]} for i in top_t],
                "t2i_top3": [{"idx": int(i), "caption": captions[i]} for i in top_i],
            }
        )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(example_rows, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    args = parse_args()
    configure_retrieval(
        max_len=args.max_len,
        batch_size=args.batch_size,
        workers=args.workers,
        num_examples=args.num_examples,
        output_dir=args.output_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SimpleTokenizer.load(args.tokenizer)

    vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    txt = TextTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_len,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_dim=1536,
        pad_id=SimpleTokenizer.PAD_ID,
    )
    model = CLIPModel(vision_encoder=vit, text_encoder=txt, embed_dim=512)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    ds = CLEVRCaptionDataset_Aa(root=args.data_root, split="val", transform=build_clip_image_transform(224))
    run_retrieval(model, ds, tokenizer, device)


if __name__ == "__main__":
    main()
