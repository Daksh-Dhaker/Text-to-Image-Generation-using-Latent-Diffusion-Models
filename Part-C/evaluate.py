import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import CLEVRDataset, build_dataloader
from scheduler import GaussianDiffusion
from unet import LatentDiffusionModel
from vae import VAE


def _load_pretrained_local_first(factory, model_name: str):
    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return factory.from_pretrained(model_name)


class FrozenCLIPTextEncoder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tokenizer = _load_pretrained_local_first(CLIPTokenizer, model_name)
        self.model = _load_pretrained_local_first(CLIPTextModel, model_name).to(device)
        self.max_length = min(
            int(getattr(self.model.config, 'max_position_embeddings', 77)),
            int(getattr(self.tokenizer, 'model_max_length', 77)),
        )
        if self.max_length <= 0 or self.max_length > 10000:
            self.max_length = 77
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        return self.model(**tokens).last_hidden_state


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t * 0.5 + 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def clear_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob('*.png'):
        item.unlink()


def load_vae(ckpt_path: str, device: torch.device) -> VAE:
    vae = VAE().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[Eval] VAE loaded from {ckpt_path}")
    return vae


def load_ldm(args, device: torch.device) -> LatentDiffusionModel:
    model = LatentDiffusionModel(
        latent_channels=4,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        time_dim=args.time_dim,
        context_dim=512,
        num_heads=args.num_heads,
        text_seq_len=77,
        gradient_checkpointing=False,
    ).to(device)
    ckpt = torch.load(args.ldm_ckpt, map_location=device)
    state = ckpt.get('model_state') or ckpt.get('unet_state') or ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load the LDM checkpoint into the current UNet. "
            "If this checkpoint was trained before the architecture fixes, retrain LDM "
            "or pass the matching --base_channels/--time_dim/--channel_multipliers."
        ) from exc
    model.eval()
    print(f"[Eval] LDM loaded from {args.ldm_ckpt}")
    return model


def load_latent_stats(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats = torch.load(path, map_location='cpu')
    mean = stats['mean'].to(device).float().view(1, -1, 1, 1)
    std = stats['std'].to(device).float().view(1, -1, 1, 1).clamp(min=1e-6)
    return mean, std


@torch.no_grad()
def eval_vae(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    real_dir = out_dir / 'real'
    recon_dir = out_dir / 'recon'
    viz_dir = out_dir / 'viz'
    clear_pngs(real_dir)
    clear_pngs(recon_dir)
    clear_pngs(viz_dir)

    vae = load_vae(args.vae_ckpt, device)
    val_loader = build_dataloader(
        args.data_root,
        'val',
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )

    idx = 0
    viz_saved = 0
    for imgs, _ in tqdm(val_loader, desc='VAE Reconstruction'):
        imgs = imgs.to(device, non_blocking=True)
        mu, _ = vae.encode(imgs)
        recon = vae.decode(mu)

        for i in range(imgs.size(0)):
            real_pil = tensor_to_pil(imgs[i])
            recon_pil = tensor_to_pil(recon[i])
            real_pil.save(real_dir / f'{idx:06d}.png')
            recon_pil.save(recon_dir / f'{idx:06d}.png')
            if viz_saved < 32:
                comparison = Image.new('RGB', (real_pil.width * 2, real_pil.height))
                comparison.paste(real_pil, (0, 0))
                comparison.paste(recon_pil, (real_pil.width, 0))
                comparison.save(viz_dir / f'compare_{viz_saved:04d}.png')
                viz_saved += 1
            idx += 1

    print(f"[VAE Eval] Saved {idx} real and reconstructed images.")
    try:
        from cleanfid import fid
    except ImportError:
        print("[VAE Eval] clean-fid not installed. Run: pip install clean-fid")
        return
    score = fid.compute_fid(str(real_dir), str(recon_dir), device=device)
    print(f"[VAE Eval] FID = {score:.4f}")
    (out_dir / 'fid.txt').write_text(f"FID (reconstruction): {score:.4f}\n")


@torch.no_grad()
def eval_ldm(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    real_dir = out_dir / 'real'
    gen_dir = out_dir / 'generated'
    viz_dir = out_dir / 'viz'
    clear_pngs(real_dir)
    clear_pngs(gen_dir)
    clear_pngs(viz_dir)

    vae = load_vae(args.vae_ckpt, device)
    model = load_ldm(args, device)
    lat_mean, lat_std = load_latent_stats(args.latent_stats, device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule='cosine').to(device)
    clip_enc = FrozenCLIPTextEncoder(args.text_encoder, device)

    val_dataset = CLEVRDataset(args.data_root, split='val', img_size=128)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    gen_idx = 0
    max_samples: Optional[int] = args.max_samples
    for imgs, captions in tqdm(val_loader, desc='LDM Generation'):
        if max_samples is not None and gen_idx >= max_samples:
            break
        if max_samples is not None:
            remaining = max_samples - gen_idx
            imgs = imgs[:remaining]
            captions = captions[:remaining]

        imgs = imgs.to(device, non_blocking=True)
        captions = list(captions)
        batch_size = imgs.size(0)

        for i in range(batch_size):
            tensor_to_pil(imgs[i]).save(real_dir / f'{gen_idx + i:06d}.png')

        context = clip_enc.encode(captions)
        null_context = model.expanded_null_context(batch_size).to(device)
        z0 = diffusion.sample(
            model=model,
            shape=(batch_size, 4, 16, 16),
            context=context,
            null_context=null_context,
            device=device,
            guidance_scale=args.guidance_scale,
            num_steps=args.sample_steps,
            eta=args.ddim_eta,
            sampler=args.sampler,
        )
        gen_imgs = vae.decode(z0 * lat_std + lat_mean)

        for i in range(batch_size):
            gen_pil = tensor_to_pil(gen_imgs[i])
            gen_pil.save(gen_dir / f'{gen_idx + i:06d}.png')
            if gen_idx + i < 32:
                real_pil = tensor_to_pil(imgs[i])
                side = Image.new('RGB', (real_pil.width * 2 + 4, real_pil.height), (128, 128, 128))
                side.paste(real_pil, (0, 0))
                side.paste(gen_pil, (real_pil.width + 4, 0))
                side.save(viz_dir / f'compare_{gen_idx + i:04d}.png')

        gen_idx += batch_size

    print(f"[LDM Eval] Saved {gen_idx} real and generated images.")
    try:
        from cleanfid import fid
    except ImportError:
        print("[LDM Eval] clean-fid not installed. Run: pip install clean-fid")
        return
    score = fid.compute_fid(str(real_dir), str(gen_dir), device=device)
    print(f"[LDM Eval] FID = {score:.4f}")
    (out_dir / 'fid.txt').write_text(f"FID (generation): {score:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    vae_parser = sub.add_parser('vae')
    vae_parser.add_argument('--data_root', required=True)
    vae_parser.add_argument('--vae_ckpt', required=True)
    vae_parser.add_argument('--output_dir', default='results/vae')
    vae_parser.add_argument('--batch_size', type=int, default=64)
    vae_parser.add_argument('--workers', type=int, default=4)

    ldm_parser = sub.add_parser('ldm')
    ldm_parser.add_argument('--data_root', required=True)
    ldm_parser.add_argument('--vae_ckpt', required=True)
    ldm_parser.add_argument('--ldm_ckpt', required=True)
    ldm_parser.add_argument('--latent_stats', required=True)
    ldm_parser.add_argument('--output_dir', default='results/ldm')
    ldm_parser.add_argument('--batch_size', type=int, default=16)
    ldm_parser.add_argument('--workers', type=int, default=4)
    ldm_parser.add_argument('--timesteps', type=int, default=500)
    ldm_parser.add_argument('--guidance_scale', type=float, default=4.0)
    ldm_parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    ldm_parser.add_argument('--sample_steps', type=int, default=500)
    ldm_parser.add_argument('--ddim_eta', type=float, default=0.0)
    ldm_parser.add_argument('--max_samples', type=int, default=None)
    ldm_parser.add_argument('--text_encoder', default='openai/clip-vit-base-patch32')
    ldm_parser.add_argument('--base_channels', type=int, default=128)
    ldm_parser.add_argument('--time_dim', type=int, default=512)
    ldm_parser.add_argument('--num_heads', type=int, default=8)
    ldm_parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4])

    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.mode == 'vae':
        eval_vae(parsed_args)
    else:
        eval_ldm(parsed_args)
