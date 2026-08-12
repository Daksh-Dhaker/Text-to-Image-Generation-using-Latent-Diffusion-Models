"""
Usage (Stage 1):
    python -m part_b.eval_vlm \
        --stage stage1 \
        --data-root /path/to/Part_A \
        --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
        --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
        --output-dir ./outputs/vlm_eval_stage1

Usage (Stage 2):
    python -m part_b.eval_vlm \
        --stage stage2 \
        --data-root /path/to/Part_A \
        --vit-ckpt ./outputs/vlm_stage1/vit_clean.pt --vit-kind raw \
        --stage1-ckpt ./outputs/vlm_stage1/checkpoints/best.pt \
        --stage2-ckpt ./outputs/vlm_stage2/checkpoints/best \
        --output-dir ./outputs/vlm_eval_stage2
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path
import random, numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models.vit import VisionTransformer
from dataset_vlm import CaptioningDataset, QADataset,_STAGE1_SYSTEM, _STAGE2_SYSTEM, _IMAGE_PLACEHOLDER
from models.projector import ReverseBNProjector
from models.vlm import VLMModel, merge_visual_tokens

from transformers import AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


def build_vlm_image_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _ensure_vit_snapshot(vit_ckpt, vit_kind, output_dir):
    if vit_kind == "raw":
        return vit_ckpt
    from train_stage1 import save_vit_snapshot
    clean = output_dir / "vit_clean.pt"
    if not clean.exists():
        save_vit_snapshot(vit_ckpt, vit_kind, clean)
    return str(clean)


def build_eval_model(stage, vit_ckpt, qwen_name, stage1_ckpt, stage2_ckpt, image_token_id, device):
    projector = ReverseBNProjector(in_dim=384, hidden_dim=1536, out_dim=2560)
    model = VLMModel(
        vit_checkpoint_path=vit_ckpt,
        projector=projector,
        qwen_model_name=qwen_name,
        image_token_id=image_token_id,
    )

    s1 = torch.load(stage1_ckpt, map_location="cpu")
    projector_sd = s1.get("projector", s1)
    model.projector.load_state_dict(projector_sd)

    if stage == "stage2":
        if stage2_ckpt is None:
            raise ValueError("--stage2-ckpt is required when --stage stage2")
        if not _HAS_PEFT:
            raise RuntimeError("peft is required to load Stage-2 LoRA adapters.")

        stage2_dir = Path(stage2_ckpt)
        lora_dir = stage2_dir / "lora"
        state_path = stage2_dir / "state.pt"

        if state_path.exists():
            s2 = torch.load(state_path, map_location="cpu")
            if "projector" in s2:
                model.projector.load_state_dict(s2["projector"])

        if lora_dir.is_dir():
            model.qwen = PeftModel.from_pretrained(model.qwen, str(lora_dir))
        elif (stage2_dir / "lora_adapter.pt").exists():
            lora_sd = torch.load(stage2_dir / "lora_adapter.pt", map_location="cpu")
            model.qwen.load_state_dict(lora_sd, strict=False)
        else:
            raise FileNotFoundError(f"No LoRA adapter found at {stage2_dir}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)

def _build_stage1_prompt() -> str:
    return (
        "<|im_start|>system\n"
        f"{_STAGE1_SYSTEM}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{_IMAGE_PLACEHOLDER}\n"
        "Describe the image."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_stage2_prompt(question) -> str:
    return (
        f"<|im_start|>system\n"
        f"{_STAGE2_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{_IMAGE_PLACEHOLDER}\n"
        f"Question: {question}\n<|im_end|>\n"
        f"<|im_start|>assistant\nReasoning: "  # matches training prefix in _collate_stage2
    )


@torch.no_grad()
def generate_response(model, image, prompt_text, tokenizer, max_new_tokens= 200, device = None):
    if device is None:
        device = next(model.parameters()).device

    if image.dim() == 3:
        image = image.unsqueeze(0) # (1, 3, H, W)
    image = image.to(device)

    visual_tokens = model.encode_image(image) # (1, 196, D_qwen)
    qwen_dtype = next(model.qwen.parameters()).dtype
    visual_tokens = visual_tokens.to(dtype=qwen_dtype)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device) # (1, T)

    embed_fn = model.qwen.get_input_embeddings()
    text_embeds = embed_fn(prompt_ids).to(dtype=qwen_dtype)  # (1, T, D)

    attn_mask = torch.ones(prompt_ids.shape, dtype=torch.long, device=device) # (1, T)

    image_token_id = model.image_token_id
    if image_token_id is not None and (prompt_ids == image_token_id).any():
        idx = int(
            (prompt_ids[0] == image_token_id)
            .nonzero(as_tuple=False)[0, 0]
        )
        dummy_labels = torch.full_like(prompt_ids, -100)# merge_visual_tokens requires labels — use dummy -100 (not supervised)
        merged_embeds, merged_attn, _ = merge_visual_tokens(
            text_embeds=text_embeds,
            image_tokens=visual_tokens,
            attention_mask=attn_mask,
            labels=dummy_labels,
            placeholder_idx=idx,
        )
    else:
        vis_attn = torch.ones(1, visual_tokens.shape[1], dtype=torch.long, device=device)
        merged_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        merged_attn   = torch.cat([vis_attn, attn_mask], dim=1)

    gen_ids = model.qwen.generate(
        inputs_embeds=merged_embeds,
        attention_mask=merged_attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize_text(s):
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_exact_match(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return 0.0
    hits = sum(int(_normalize_text(p) == _normalize_text(r)) for p, r in zip(predictions, references))
    return hits / len(predictions)


def compute_bleu(predictions, references):
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("sacrebleu is required for BLEU. `pip install sacrebleu`.") from e
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def extract_answer_from_cot(text):
    m = re.search(r"answer\s*[:\-]\s*(.+?)(?:[\.\n]|$)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def save_qualitative_pdf(records, images_root, save_path, n_correct=8, n_incorrect=8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        raise ImportError("matplotlib is required for qualitative PDF output.") from e

    save_path.parent.mkdir(parents=True, exist_ok=True)

    correct = [r for r in records if r.get("correct")][:n_correct]
    incorrect = [r for r in records if not r.get("correct")][:n_incorrect]

    def _render_group(pdf, group, title_prefix):
        for r in group:
            fname = r.get("image_filename")
            img_path = (Path(images_root) / fname) if fname else None
            fig, ax = plt.subplots(figsize=(8.5, 6))
            try:
                if img_path is None or not img_path.exists():
                    raise FileNotFoundError
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, f"[missing image]\n{img_path}", ha="center", va="center")
            ax.axis("off")
            caption = (
                f"{title_prefix}\n"
                f"Prompt: {r.get('prompt', '')}\n\n"
                f"Prediction:\n{r.get('prediction', '')}\n\n"
                f"Gold:\n{r.get('reference', '')}"
            )
            fig.suptitle(caption, fontsize=8, wrap=True)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    with PdfPages(str(save_path)) as pdf:
        _render_group(pdf, correct, "CORRECT")
        _render_group(pdf, incorrect, "INCORRECT")
    print(f"[save_qualitative_pdf] wrote {save_path}")


@torch.no_grad()
def run_eval(model, dataset, tokenizer, device,
             stage="stage1", max_new_tokens=200, limit=None, images_root=None,
             output_dir=None, log_every=50):
    model.eval()

    preds, refs, records = [], [], []
    n = len(dataset) if limit is None else min(limit, len(dataset))

    for i in range(n):
        item = dataset[i]

        if stage == "stage1":
            image, caption = item[0], item[1]
            prompt = _build_stage1_prompt()
            gold = caption
            out_txt = generate_response(model, image, prompt, tokenizer,
                                        max_new_tokens=max_new_tokens, device=device)
            pred = out_txt.strip()
            is_correct = (_normalize_text(pred) == _normalize_text(gold))
            fname = None
            if hasattr(dataset, "samples") and i < len(dataset.samples):
                s = dataset.samples[i]
                fname = s[0] if isinstance(s, (list, tuple)) else None
            records.append({
                "idx": i,
                "image_filename": fname,
                "prompt": prompt,
                "prediction": pred,
                "reference": gold,
                "correct": is_correct,
            })
            preds.append(pred)
            refs.append(gold)

        elif stage == "stage2":
            image, question, explanation, answer = (item[0], item[1], item[2], item[3])
            fname = None
            if hasattr(dataset, "samples") and i < len(dataset.samples):
                s = dataset.samples[i]
                fname = s.get("image_filename") if isinstance(s, dict) else None

            prompt = _build_stage2_prompt(question)
            out_txt = generate_response(model, image, prompt, tokenizer,
                                        max_new_tokens=max_new_tokens, device=device)
            pred_answer = extract_answer_from_cot(out_txt)
            is_correct = (_normalize_text(pred_answer) == _normalize_text(str(answer)))
            records.append({
                "idx": i,
                "image_filename": fname,
                "prompt": prompt,
                "prediction": out_txt.strip(),
                "predicted_answer": pred_answer,
                "reference": str(answer),
                "reference_explanation": explanation,
                "correct": is_correct,
            })
            preds.append(pred_answer)
            refs.append(str(answer))

        else:
            raise ValueError(f"Unknown stage: {stage}")

        if (i + 1) % log_every == 0:
            running_em = sum(r["correct"] for r in records) / len(records)
            print(f"  [{i+1}/{n}] running EM = {running_em:.4f}", flush=True)

    metrics = {
        "n_examples": len(preds),
        "exact_match": compute_exact_match(preds, refs),
    }
    if stage == "stage1":
        metrics["bleu"] = compute_bleu(preds, refs)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<14}: {v:.4f}")
        else:
            print(f"  {k:<14}: {v}")
    print("==========================\n",flush=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "records.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if images_root is not None:
            save_qualitative_pdf(records, images_root, output_dir / "qualitative.pdf")

    return metrics

def parse_args():
    p = argparse.ArgumentParser(description="Part B evaluation")
    p.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True)
    p.add_argument("--data-root", type=str, required=True,
                   help="Root of Part-A data (contains Clevr_official/ and "
                        "Probe-Datasets/).")
    p.add_argument("--train-images", type=str, default=None)
    p.add_argument("--val-images", type=str, default=None)

    p.add_argument("--qwen-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vit-ckpt", type=str, required=True)
    p.add_argument("--vit-kind", type=str, default="raw",
                   choices=["clip_vit", "dino_student", "dino_teacher", "raw"])
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument("--stage2-ckpt", type=str, default=None)

    p.add_argument("--output-dir", type=str, default="./outputs/vlm_eval")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--n-correct", type=int, default=8)
    p.add_argument("--n-incorrect", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_clean = _ensure_vit_snapshot(args.vit_ckpt, args.vit_kind, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_name, trust_remote_code=True)#NEED TO CHEECK , AS IT Requires Web
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>"]}
    )
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print(f"[main] <image> token id = {image_token_id}",flush=True)

    model = build_eval_model(
        stage=args.stage,
        vit_ckpt=vit_clean,
        qwen_name=args.qwen_name,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        image_token_id=image_token_id,
        device=device,
    )

    transform = build_vlm_image_transform(224)
    all_metrics = {}

    if args.stage == "stage1":
        train_images = (Path(args.train_images) if args.train_images else Path(args.data_root) / "Clevr_official" / "images" / "train")
        val_images = (Path(args.val_images)   if args.val_images else Path(args.data_root) / "Clevr_official" / "images" / "val")
        train_ds = CaptioningDataset(root=args.data_root, split="train", transform=transform)
        val_ds = CaptioningDataset(root=args.data_root, split="val", transform=transform)

        print("[eval] Running Stage-1 on validation ...", flush=True)
        val_metrics = run_eval(
            model, val_ds, tokenizer, device,
            stage="stage1", max_new_tokens=args.max_new_tokens,
            limit=args.limit_val,
            images_root=val_images, output_dir=output_dir / "val",
        )
        all_metrics["val"] = val_metrics

        print("[eval] Running Stage-1 on training (subset) ...", flush=True)
        train_metrics = run_eval(
            model, train_ds, tokenizer, device,
            stage="stage1", max_new_tokens=args.max_new_tokens,
            limit=args.limit_train if args.limit_train else min(2000, len(train_ds)),
            images_root=train_images, output_dir=output_dir / "train",
        )
        all_metrics["train"] = train_metrics

    else: # stage2
        val_images = (Path(args.val_images)   if args.val_images else Path(args.data_root) / "Clevr_official" / "images" / "val")
        train_images = (Path(args.train_images) if args.train_images else Path(args.data_root) / "Clevr_official" / "images" / "train")

        val_ds = QADataset(
            root=args.data_root, split="val",
            transform=transform, training=False,
            max_samples=6000,
        )
        print("[eval] Running Stage-2 on validation ...",flush=True)
        val_metrics = run_eval(
            model, val_ds, tokenizer, device,
            stage="stage2", max_new_tokens=args.max_new_tokens,
            limit=args.limit_val,
            images_root=val_images, output_dir=output_dir / "val",
        )
        all_metrics["val"] = val_metrics

        train_ds = QADataset(
            root=args.data_root, split="train",
            transform=transform, training=False,
            max_samples=6000,
        )
        print("[eval] Running Stage-2 on training (subset) ...", flush=True)
        train_metrics = run_eval(
            model, train_ds, tokenizer, device,
            stage="stage2", max_new_tokens=args.max_new_tokens,
            limit=(args.limit_train if args.limit_train else min(2000, len(train_ds))),
            images_root=train_images, output_dir=output_dir / "train",
        )
        all_metrics["train"] = train_metrics

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()