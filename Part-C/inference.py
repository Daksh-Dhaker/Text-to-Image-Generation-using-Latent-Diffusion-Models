import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from scheduler import GaussianDiffusion

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from evaluate import FrozenCLIPTextEncoder, load_latent_stats, load_ldm, load_vae, tensor_to_pil


def install_safe_torch_load() -> None:
    original_load = torch.load

    def safe_load(*args, **kwargs):
        if "weights_only" in kwargs:
            return original_load(*args, **kwargs)
        try:
            return original_load(*args, **kwargs, weights_only=True)
        except Exception as exc:
            message = str(exc)
            if "Weights only load failed" in message or "WeightsUnpickler error" in message:
                safe_globals = getattr(torch.serialization, "safe_globals", None)
                if safe_globals is not None:
                    try:
                        with safe_globals([np.core.multiarray.scalar]):
                            return original_load(*args, **kwargs, weights_only=True)
                    except Exception:
                        pass
                print("[Inference] WARNING: Falling back to unsafe torch.load (weights_only=False).")
                return original_load(*args, **kwargs, weights_only=False)
            raise

    torch.load = safe_load


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = base.replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not base:
        return ""
    if not base.lower().endswith(".png"):
        base = f"{base}.png"
    return base


def ensure_dir_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def _extract_caption(item: Dict[str, object]) -> str:
    for key in ("caption", "caption_text", "text", "sentence"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    nested = item.get("captions")
    if isinstance(nested, list) and nested:
        first = nested[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("caption") or first.get("text") or "")
    return ""


def _extract_name(item: Dict[str, object]) -> Optional[str]:
    for key in (
        "image",
        "image_filename",
        "file_name",
        "filename",
        "image_name",
        "image_path",
    ):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def records_from_json(path: Path) -> List[Dict[str, Optional[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        for key in ("annotations", "captions", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break

    if isinstance(data, dict):
        records = []
        for key in sorted(data.keys()):
            value = data[key]
            caption = ""
            if isinstance(value, dict):
                caption = _extract_caption(value)
            elif isinstance(value, str):
                caption = value
            records.append({"text": caption, "name": key})
        return records

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return [{"text": item, "name": None} for item in data]
        if isinstance(data[0], dict):
            records = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                caption = _extract_caption(item)
                name = _extract_name(item)
                records.append({"text": caption, "name": name})
            return records

    raise ValueError(f"Unrecognized JSON format in {path}")


def records_from_jsonl(path: Path) -> List[Dict[str, Optional[str]]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, str):
                records.append({"text": item, "name": None})
            elif isinstance(item, dict):
                records.append({"text": _extract_caption(item), "name": _extract_name(item)})
    return records


def load_caption_records(path: Path) -> List[Dict[str, Optional[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Caption JSON not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records = records_from_jsonl(path)
    else:
        records = records_from_json(path)
    if not records:
        raise ValueError(f"No captions found in {path}")
    return records


def batched(items: List[Dict[str, Optional[str]]], batch_size: int) -> Iterable[List[Dict[str, Optional[str]]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def find_model_file(model_dir: Path, candidates: List[str], patterns: List[str], label: str) -> Path:
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    for pattern in patterns:
        matches = sorted(model_dir.rglob(pattern))
        if matches:
            return matches[0]
    tried = candidates + patterns
    raise FileNotFoundError(f"Could not find {label} in {model_dir}. Tried: {tried}")


def find_text_encoder_dir(model_dir: Path) -> Optional[Path]:
    candidates = [
        "text_encoder",
        "clip",
        "clip_text",
        "clip-vit-base-patch32",
        "openai-clip-vit-base-patch32",
    ]
    for name in candidates:
        path = model_dir / name
        if (path / "config.json").exists():
            return path

    for config_path in model_dir.rglob("config.json"):
        parent = config_path.parent
        if (parent / "tokenizer.json").exists() or (parent / "vocab.json").exists():
            return parent
    return None


def list_image_paths(image_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def load_image_tensor(path: Path, img_size: int = 128) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def iter_image_batches(paths: List[Path], batch_size: int, img_size: int = 128):
    for batch_paths in batched(paths, batch_size):
        images = [load_image_tensor(p, img_size=img_size) for p in batch_paths]
        yield batch_paths, torch.stack(images, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part C LDM inference")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--task", choices=["reconstruct", "generate"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_safe_torch_load()

    model_dir = Path(args.model_dir)
    ensure_dir_exists(model_dir, "Model directory")

    output_dir = Path(args.output_dir)
    ensure_dir_exists(output_dir, "Output directory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] device={device}")

    vae_ckpt = find_model_file(
        model_dir,
        candidates=["vae_best.pt", "vae_best_fid.pt", "vae.pt", "vae_ckpt.pt"],
        patterns=["**/vae_best.pt", "**/vae_best_fid.pt", "**/vae*.pt"],
        label="VAE checkpoint",
    )
    vae = load_vae(str(vae_ckpt), device)

    if args.task == "reconstruct":
        image_dir = Path(args.data_path)
        ensure_dir_exists(image_dir, "Input image directory")
        image_paths = list_image_paths(image_dir)
        batch_size = 16

        total = 0
        with torch.no_grad():
            for batch_paths, batch in iter_image_batches(image_paths, batch_size, img_size=128):
                batch = batch.to(device, non_blocking=True)
                mu, _ = vae.encode(batch)
                recon = vae.decode(mu)
                for src_path, image in zip(batch_paths, recon):
                    tensor_to_pil(image).save(output_dir / src_path.name)
                total += len(batch_paths)

        print(f"[Inference] Saved {total} reconstructions to {output_dir}")
        return

    ldm_ckpt = find_model_file(
        model_dir,
        candidates=["ldm_best.pt", "ldm.pt", "ldm_ckpt.pt", "unet.pt", "unet_best.pt"],
        patterns=["**/ldm_best.pt", "**/ldm*.pt", "**/unet*.pt"],
        label="LDM checkpoint",
    )
    latent_stats = find_model_file(
        model_dir,
        candidates=["latent_stats.pt"],
        patterns=["**/latent_stats.pt"],
        label="latent stats",
    )
    text_encoder_dir = find_text_encoder_dir(model_dir)
    if text_encoder_dir is None:
        raise FileNotFoundError(
            "CLIP text encoder files not found in model_dir. "
            "Expected a folder like model_dir/text_encoder or model_dir/clip containing config.json and tokenizer files."
        )

    ldm_args = SimpleNamespace(
        ldm_ckpt=str(ldm_ckpt),
        base_channels=128,
        time_dim=512,
        num_heads=8,
        channel_multipliers=[1, 2, 4],
    )
    model = load_ldm(ldm_args, device)
    lat_mean, lat_std = load_latent_stats(str(latent_stats), device)
    diffusion = GaussianDiffusion(timesteps=500, schedule="cosine").to(device)
    clip_enc = FrozenCLIPTextEncoder(str(text_encoder_dir), device)

    caption_path = Path(args.data_path)
    records = load_caption_records(caption_path)
    batch_size = 32

    total = 0
    with torch.no_grad():
        for batch in batched(records, batch_size):
            texts = [item.get("text") or "" for item in batch]
            names = [item.get("name") for item in batch]
            context = clip_enc.encode(texts)
            null_context = model.expanded_null_context(len(batch)).to(device)

            z0 = diffusion.sample(
                model=model,
                shape=(len(batch), 4, 16, 16),
                context=context,
                null_context=null_context,
                device=device,
                guidance_scale=4.0,
                num_steps=500,
                eta=0.0,
                sampler="ddpm",
            )
            images = vae.decode(z0 * lat_std + lat_mean)

            for i, image in enumerate(images):
                name = names[i]
                if name:
                    name = safe_filename(name)
                if not name:
                    name = f"sample_{total + i:06d}.png"
                tensor_to_pil(image).save(output_dir / name)

            total += len(batch)

    print(f"[Inference] Saved {total} generated images to {output_dir}")


if __name__ == "__main__":
    main()
