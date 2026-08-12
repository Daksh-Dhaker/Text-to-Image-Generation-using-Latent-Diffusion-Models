import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from vae import VAE
from unet import LatentDiffusionModel
from scheduler import GaussianDiffusion
from dataset import build_dataloader, CLEVRDataset

try:
    from torch.amp import GradScaler, autocast

    def make_scaler(enabled: bool):
        return GradScaler('cuda', enabled=enabled)

    def amp_context(device: torch.device, enabled: bool):
        return autocast(device_type=device.type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def make_scaler(enabled: bool):
        return GradScaler(enabled=enabled)

    def amp_context(device: torch.device, enabled: bool):
        return autocast(enabled=enabled)






class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, device, model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = self._load_pretrained(CLIPTokenizer, model_name)
        self.model     = self._load_pretrained(CLIPTextModel, model_name).to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.model.eval()

    @staticmethod
    def _load_pretrained(factory, model_name: str):
        try:
            return factory.from_pretrained(model_name, local_files_only=True)
        except OSError:
            return factory.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 77) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt',
        ).to(self.device)
        return self.model(**tokens).last_hidden_state  

    def forward(self, texts):
        return self.encode(texts)






@torch.no_grad()
def compute_latent_stats(vae, loader, device):
    print("[LDM] Computing latent statistics over training set …")
    all_mu = []
    for imgs, _ in tqdm(loader, desc='  encoding'):
        mu, _ = vae.encode(imgs.to(device))
        all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)            
    mean   = all_mu.mean(dim=(0, 2, 3))          
    std    = all_mu.std(dim=(0, 2, 3)).clamp(min=1e-6)
    print(f"[LDM] Latent mean={mean.tolist()}  std={std.tolist()}")
    return mean, std


def lat_to_device(mean, std, device):

    return (
        mean.to(device).view(1, -1, 1, 1),
        std.to(device).view(1, -1, 1, 1),
    )


class LatentCacheDataset(Dataset):


    def __init__(self, cache_path: Path, lat_mean=None, lat_std=None):
        payload = torch.load(cache_path, map_location='cpu')
        self.latents = payload['latents']
        self.captions = payload['captions']
        self.text_embeddings = payload.get('text_embeddings')
        self.lat_mean = lat_mean.float().view(-1, 1, 1).cpu() if lat_mean is not None else None
        self.lat_std = lat_std.float().view(-1, 1, 1).cpu().clamp(min=1e-6) if lat_std is not None else None

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        latent = self.latents[idx].float()
        if self.lat_mean is not None and self.lat_std is not None:
            latent = (latent - self.lat_mean) / self.lat_std
        item = {'latent': latent, 'caption': self.captions[idx]}
        if self.text_embeddings is not None:
            item['context'] = self.text_embeddings[idx].float()
        return item


def make_cache_path(cache_dir: Path, split: str, cache_text: bool) -> Path:
    suffix = 'latents_text' if cache_text else 'latents'
    return cache_dir / f'{split}_{suffix}.pt'


@torch.no_grad()
def build_or_load_feature_cache(args, split: str, vae, clip_enc, device, cache_dir: Path):
    if not args.cache_latents:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = make_cache_path(cache_dir, split, args.cache_text)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"[Cache] Using {split} cache: {cache_path}")
        return cache_path

    print(f"[Cache] Building {split} latent cache at {cache_path}")
    dataset = CLEVRDataset(args.data_root, split=split, img_size=128)
    loader = DataLoader(
        dataset,
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )

    latent_chunks = []
    text_chunks = []
    captions_all = []
    cache_dtype = torch.float16 if args.cache_dtype == 'fp16' else torch.float32
    vae.eval()
    for imgs, captions in tqdm(loader, desc=f'  cache {split}'):
        imgs = imgs.to(device, non_blocking=True)
        mu, _ = vae.encode(imgs)
        latent_chunks.append(mu.detach().cpu().to(cache_dtype))
        captions = list(captions)
        captions_all.extend(captions)
        if args.cache_text:
            context = clip_enc.encode(captions)
            text_chunks.append(context.detach().cpu().to(cache_dtype))

    payload = {
        'latents': torch.cat(latent_chunks, dim=0),
        'captions': captions_all,
        'cache_text': args.cache_text,
        'cache_dtype': args.cache_dtype,
    }
    if text_chunks:
        payload['text_embeddings'] = torch.cat(text_chunks, dim=0)
    torch.save(payload, cache_path)
    return cache_path


def compute_latent_stats_from_cache(cache_path: Path):
    payload = torch.load(cache_path, map_location='cpu')
    latents = payload['latents'].float()
    mean = latents.mean(dim=(0, 2, 3))
    std = latents.std(dim=(0, 2, 3), unbiased=False).clamp(min=1e-6)
    print(f"[LDM] Latent mean={mean.tolist()}  std={std.tolist()}")
    return mean, std


def build_cached_loader(cache_path: Path, split: str, args, lat_mean, lat_std):
    dataset = LatentCacheDataset(cache_path, lat_mean=lat_mean, lat_std=lat_std)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(args.workers > 0),
    )


@torch.no_grad()
def batch_to_latents_and_context(batch, using_cache, vae, clip_enc, lat_mean_d, lat_std_d, device):
    if using_cache:
        z0 = batch['latent'].to(device, non_blocking=True)
        if 'context' in batch:
            context = batch['context'].to(device, non_blocking=True)
        else:
            context = clip_enc.encode(list(batch['caption']))
        return z0, context

    imgs, captions = batch
    imgs = imgs.to(device, non_blocking=True)
    mu, _ = vae.encode(imgs)
    z0 = (mu - lat_mean_d) / lat_std_d
    context = clip_enc.encode(list(captions))
    return z0, context






def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t   = (t * 0.5 + 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(message: str) -> None:
    print(message, flush=True)


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)






@torch.no_grad()
def compute_fid_ldm(
    model, vae, diffusion, clip_enc,
    val_loader, lat_mean, lat_std,
    device, fid_tmp_dir, epoch, fid_log_path,
    guidance_scale=4.0, fid_samples=1024, ddim_steps=50, sampler='ddim',
):
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[FID] clean-fid not installed — skipping.")
        return None

    real_dir = fid_tmp_dir / 'real'
    gen_dir  = fid_tmp_dir / 'generated'
    real_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    for f in real_dir.glob('*.png'): f.unlink()
    for f in gen_dir.glob('*.png'):  f.unlink()

    model.eval()
    gen_idx = 0

    for imgs, captions in tqdm(val_loader, desc=f'  FID gen (epoch {epoch})'):
        if gen_idx >= fid_samples:
            break
        B        = imgs.size(0)
        imgs     = imgs.to(device)
        captions = list(captions)

        context   = clip_enc.encode(captions)                    
        null_ctx  = model.expanded_null_context(B).to(device)   

        z0 = diffusion.sample(
            model=model,
            shape=(B, 4, 16, 16),
            context=context,
            null_context=null_ctx,
            device=device,
            guidance_scale=guidance_scale,
            num_steps=ddim_steps,
            sampler=sampler,
        )
        z0      = z0 * lat_std + lat_mean     
        gen_imgs = vae.decode(z0)             

        for i in range(B):
            tensor_to_pil(imgs[i]).save(real_dir / f'{gen_idx:06d}.png')
            tensor_to_pil(gen_imgs[i]).save(gen_dir / f'{gen_idx:06d}.png')
            gen_idx += 1

    model.train()

    score  = cleanfid.compute_fid(str(real_dir), str(gen_dir), device=device, verbose=False)
    banner = '=' * 60
    print(f"\n{banner}")
    print(f"  [FID] Epoch {epoch:4d}  LDM Generation FID = {score:.4f}  (n={gen_idx}  sampler={sampler}  steps={ddim_steps})")
    print(f"{banner}\n")

    with open(fid_log_path, 'a') as f:
        f.write(json.dumps({'epoch': epoch, 'fid': round(score, 4), 'n': gen_idx, 'sampler': sampler}) + '\n')

    return score






@torch.no_grad()
def save_sample_grid(model, vae, diffusion, clip_enc, lat_mean, lat_std,
                     device, viz_dir, epoch, guidance_scale, ddim_steps=50, sampler='ddim', n=4):
    model.eval()
    prompts = [
        "An image with 2 objects: 1 large red metal cube, 1 small blue rubber sphere",
        "An image with 3 objects: 1 large green rubber sphere, 1 small yellow metal cube, 1 large cyan rubber cylinder",
        "An image with 1 objects: 1 large purple metal sphere",
        "An image with 4 objects: 2 large red metal cubes, 2 small blue rubber spheres",
    ][:n]

    ctx      = clip_enc.encode(prompts)
    null_ctx = model.expanded_null_context(n).to(device)

    z0 = diffusion.sample(
        model=model, shape=(n, 4, 16, 16),
        context=ctx, null_context=null_ctx,
        device=device, guidance_scale=guidance_scale,
        num_steps=ddim_steps,
        sampler=sampler,
    )
    z0   = z0 * lat_std + lat_mean
    imgs = (vae.decode(z0) * 0.5 + 0.5).clamp(0, 1)
    save_image(imgs, viz_dir / f'sample_epoch{epoch:04d}.png', nrow=n)
    print(f"  [Viz] {viz_dir}/sample_epoch{epoch:04d}.png")
    model.train()






def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[LDM Training] device={device}")

    save_dir = Path(args.save_dir)
    viz_dir  = save_dir / 'generated_viz'
    fid_tmp  = save_dir / 'fid_tmp'
    fid_log  = save_dir / 'fid_log.jsonl'
    metrics_csv = save_dir / 'metrics.csv'
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    fid_tmp.mkdir(exist_ok=True)

    
    vae  = VAE().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt.get('model_state', ckpt))
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f"[LDM] VAE loaded from {args.vae_ckpt}")

    
    clip_enc = FrozenCLIPTextEncoder(device, model_name=args.text_encoder)

    
    cache_dir = Path(args.cache_dir) if args.cache_dir else save_dir / 'feature_cache'
    train_cache = build_or_load_feature_cache(args, 'train', vae, clip_enc, device, cache_dir)
    val_cache = build_or_load_feature_cache(args, 'val', vae, clip_enc, device, cache_dir)
    using_cache = train_cache is not None and val_cache is not None

    
    stats_path = save_dir / 'latent_stats.pt'
    if stats_path.exists() and not args.recompute_stats and not args.rebuild_cache:
        s        = torch.load(stats_path, map_location='cpu')
        lat_mean = s['mean']   
        lat_std  = s['std']
        print(f"[LDM] Loaded latent stats from {stats_path}")
    elif train_cache is not None:
        lat_mean, lat_std = compute_latent_stats_from_cache(train_cache)
        torch.save({'mean': lat_mean.cpu(), 'std': lat_std.cpu()}, stats_path)
    else:
        stats_loader = build_dataloader(
            args.data_root,
            'train',
            args.batch_size,
            args.workers,
            shuffle=False,
            drop_last=False,
        )
        lat_mean, lat_std = compute_latent_stats(vae, stats_loader, device)
        torch.save({'mean': lat_mean.cpu(), 'std': lat_std.cpu()}, stats_path)

    lat_mean_d, lat_std_d = lat_to_device(lat_mean, lat_std, device)
    lat_mean_list = [float(x) for x in lat_mean.cpu().tolist()]
    lat_std_list = [float(x) for x in lat_std.cpu().tolist()]
    metrics_fields = [
        'epoch',
        'train_loss', 'val_loss', 'lr',
        'fid', 'fid_samples', 'fid_steps', 'fid_sampler', 'guidance_scale',
        'lat_mean_0', 'lat_mean_1', 'lat_mean_2', 'lat_mean_3',
        'lat_std_0', 'lat_std_1', 'lat_std_2', 'lat_std_3',
    ]

    
    if using_cache:
        train_loader = build_cached_loader(train_cache, 'train', args, lat_mean, lat_std)
        val_loader = build_cached_loader(val_cache, 'val', args, lat_mean, lat_std)
        print(f"[DataLoader] cached train batches={len(train_loader)}  val batches={len(val_loader)}")
    else:
        train_loader = build_dataloader(args.data_root, 'train', args.batch_size, args.workers)
        val_loader = build_dataloader(args.data_root, 'val', args.batch_size, args.workers, shuffle=False, drop_last=False)

    fid_val_loader = build_dataloader(
        args.data_root,
        'val',
        args.batch_size,
        args.workers,
        shuffle=False,
        drop_last=False,
    )

    
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule='cosine').to(device)

    
    model = LatentDiffusionModel(
        latent_channels=4,
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        time_dim=args.time_dim,
        context_dim=512,
        num_heads=args.num_heads,
        text_seq_len=77,
        gradient_checkpointing=args.grad_checkpoint,
    ).to(device)

    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))

    updates_per_epoch = int(np.ceil(len(train_loader) / max(args.grad_accum_steps, 1)))
    total_steps   = max(updates_per_epoch * args.epochs, 1)
    warmup_steps  = args.warmup_steps if args.warmup_steps > 0 else updates_per_epoch * 5
    warmup_steps  = min(warmup_steps, max(total_steps - 1, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled  = device.type == 'cuda'
    scaler       = make_scaler(amp_enabled)

    best_val_loss = float('inf')
    best_fid      = float('inf')
    cfg_dropout   = args.cfg_dropout

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        log(
            f"[Train] Epoch {epoch}/{args.epochs} starting: "
            f"{n_batches} batches, batch_size={args.batch_size}, workers={args.workers}"
        )
        log("[Train] Waiting for first batch from DataLoader...")

        for step, batch in enumerate(train_loader):
            if step == 0:
                log("[Train] First batch loaded; training step has started.")
            z0, context = batch_to_latents_and_context(
                batch, using_cache, vae, clip_enc, lat_mean_d, lat_std_d, device
            )
            B = z0.size(0)

            
            t     = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(z0)
            zt    = diffusion.q_sample(z0, t, noise)

            
            null_ctx = model.expanded_null_context(B).to(device)
            if cfg_dropout > 0:
                mask    = (torch.rand(B, device=device) < cfg_dropout)[:, None, None]
                context = torch.where(mask.expand_as(context), null_ctx, context)

            with amp_context(device, amp_enabled):
                pred_noise = model(zt, t, context)
                loss       = nn.functional.mse_loss(pred_noise, noise)

            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = (step + 1) % args.grad_accum_steps == 0 or step + 1 == n_batches
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            epoch_loss += loss.item()

            if step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                log(
                    f"Epoch {epoch:3d}/{args.epochs}  step {step:4d}/{n_batches}  "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}"
                )

        avg_loss = epoch_loss / n_batches
        print(f"[Train] Epoch {epoch}  avg_loss={avg_loss:.4f}")

        
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                z0, ctx = batch_to_latents_and_context(
                    batch, using_cache, vae, clip_enc, lat_mean_d, lat_std_d, device
                )
                B = z0.size(0)
                t     = torch.randint(0, args.timesteps, (B,), device=device)
                noise = torch.randn_like(z0)
                zt    = diffusion.q_sample(z0, t, noise)
                with amp_context(device, amp_enabled):
                    pred  = model(zt, t, ctx)
                    loss  = nn.functional.mse_loss(pred, noise)
                val_total += loss.item() * B
                val_n     += B
        val_loss = val_total / val_n
        model.train()

        print(f"[Val]   Epoch {epoch}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()},
                save_dir / 'ldm_best.pt',
            )
            print(f"  ** New best LDM saved (val_loss={val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()},
                save_dir / f'ldm_epoch{epoch:04d}.pt',
            )
            save_sample_grid(
                model, vae, diffusion, clip_enc,
                lat_mean_d, lat_std_d, device,
                viz_dir, epoch, args.guidance_scale,
                ddim_steps=args.ddim_steps,
                sampler=args.eval_sampler,
            )

        
        fid_score = None
        if args.fid_every > 0 and epoch % args.fid_every == 0:
            fid_score = compute_fid_ldm(
                model, vae, diffusion, clip_enc,
                fid_val_loader, lat_mean_d, lat_std_d,
                device, fid_tmp, epoch, fid_log,
                guidance_scale=args.guidance_scale,
                fid_samples=args.fid_samples,
                ddim_steps=args.ddim_steps,
                sampler=args.eval_sampler,
            )
            if fid_score is not None:
                if fid_score < best_fid:
                    best_fid = fid_score
                    torch.save(
                        {'epoch': epoch, 'model_state': model.state_dict(), 'fid': fid_score},
                        save_dir / 'ldm_best_fid.pt',
                    )
                    print(f"  ** New best FID checkpoint (fid={fid_score:.4f})")

        lr_now = optimizer.param_groups[0]['lr']
        append_csv_row(
            metrics_csv,
            {
                'epoch': epoch,
                'train_loss': round(avg_loss, 6),
                'val_loss': round(val_loss, 6),
                'lr': lr_now,
                'fid': '' if fid_score is None else round(fid_score, 6),
                'fid_samples': args.fid_samples,
                'fid_steps': args.ddim_steps,
                'fid_sampler': args.eval_sampler,
                'guidance_scale': args.guidance_scale,
                'lat_mean_0': round(lat_mean_list[0], 6),
                'lat_mean_1': round(lat_mean_list[1], 6),
                'lat_mean_2': round(lat_mean_list[2], 6),
                'lat_mean_3': round(lat_mean_list[3], 6),
                'lat_std_0': round(lat_std_list[0], 6),
                'lat_std_1': round(lat_std_list[1], 6),
                'lat_std_2': round(lat_std_list[2], 6),
                'lat_std_3': round(lat_std_list[3], 6),
            },
            metrics_fields,
        )

    print(f"\n[LDM Training] Done.")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print(f"  Best FID      : {best_fid:.4f}")






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',       type=str,   required=True)
    p.add_argument('--vae_ckpt',        type=str,   required=True)
    p.add_argument('--save_dir',        type=str,   default='checkpoints/ldm_v2')
    p.add_argument('--epochs',          type=int,   default=150)
    p.add_argument('--batch_size',      type=int,   default=32)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--timesteps',       type=int,   default=500)
    p.add_argument('--guidance_scale',  type=float, default=4.0)
    p.add_argument('--workers',         type=int,   default=4)
    p.add_argument('--log_every',       type=int,   default=100)
    p.add_argument('--seed',            type=int,   default=775)
    p.add_argument('--fid_every',       type=int,   default=10)
    p.add_argument('--fid_samples',     type=int,   default=1024)
    p.add_argument('--ddim_steps',      type=int,   default=50,
                   help='DDIM inference steps for FID eval (50 >> 500 DDPM steps, ~10x faster)')
    p.add_argument('--eval_sampler',    choices=['ddim', 'ddpm'], default='ddim',
                   help='Sampler for previews and inline FID. Use ddpm with 500 steps for final reporting.')
    p.add_argument('--grad_checkpoint', action='store_true',
                   help='Enable gradient checkpointing to save VRAM')
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--grad_clip_norm',  type=float, default=1.0)
    p.add_argument('--warmup_steps',    type=int, default=0,
                   help='0 means five warmup epochs after gradient accumulation.')
    p.add_argument('--min_lr_scale',    type=float, default=0.01)
    p.add_argument('--cfg_dropout',     type=float, default=0.10)
    p.add_argument('--text_encoder',    type=str, default='openai/clip-vit-base-patch32')
    p.add_argument('--base_channels',   type=int, default=128)
    p.add_argument('--time_dim',        type=int, default=512)
    p.add_argument('--num_heads',       type=int, default=8)
    p.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4])
    p.set_defaults(cache_latents=True)
    p.add_argument('--cache_latents', dest='cache_latents', action='store_true',
                   help='Cache frozen VAE latents before UNet training (default).')
    p.add_argument('--no_cache_latents', dest='cache_latents', action='store_false',
                   help='Disable latent caching and encode VAE latents on the fly.')
    p.add_argument('--cache_text',      action='store_true',
                   help='Also cache CLIP text embeddings. Faster, but can use several GB of disk/RAM.')
    p.add_argument('--cache_dir',       type=str, default=None)
    p.add_argument('--cache_batch_size', type=int, default=128)
    p.add_argument('--cache_dtype',     choices=['fp16', 'fp32'], default='fp16')
    p.add_argument('--rebuild_cache',   action='store_true')
    p.add_argument('--recompute_stats', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
