

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from vae import VAE, vae_loss
from dataset import build_dataloader

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






class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, backup: dict):
        for name, param in self.model.named_parameters():
            param.data.copy_(backup[name])

    def backup(self) -> dict:
        return {k: v.clone() for k, v in self.model.named_parameters()}






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
def validate(model, loader, device, kl_weight, viz_dir, epoch):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    n = 0
    first_batch_saved = False

    for imgs, _ in loader:
        imgs = imgs.to(device)
        with amp_context(device, device.type == 'cuda'):
            recon, mu, logvar = model(imgs)
            loss, r_loss, kl  = vae_loss(recon, imgs, mu, logvar, kl_weight)

        B = imgs.size(0)
        total_loss  += loss.item()  * B
        total_recon += r_loss.item()* B
        total_kl    += kl.item()   * B
        n           += B

        if not first_batch_saved:
            grid = torch.cat([imgs[:8], recon[:8]], dim=0)
            save_image(grid * 0.5 + 0.5, viz_dir / f'recon_epoch{epoch:04d}.png', nrow=8)
            first_batch_saved = True

    model.train()
    return total_loss / n, total_recon / n, total_kl / n






@torch.no_grad()
def compute_fid_vae(model, loader, device, fid_tmp_dir, epoch, fid_log_path):

    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[FID] clean-fid not installed — skipping. Run: pip install clean-fid")
        return None

    real_dir  = fid_tmp_dir / 'real'
    recon_dir = fid_tmp_dir / 'recon'
    real_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    
    for f in real_dir.glob('*.png'):
        f.unlink()
    for f in recon_dir.glob('*.png'):
        f.unlink()

    model.eval()
    idx = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        mu, _ = model.encode(imgs)
        recon  = model.decode(mu)
        for i in range(imgs.size(0)):
            tensor_to_pil(imgs[i]).save(real_dir  / f'{idx:06d}.png')
            tensor_to_pil(recon[i]).save(recon_dir / f'{idx:06d}.png')
            idx += 1
    model.train()

    score = cleanfid.compute_fid(str(real_dir), str(recon_dir), device=device, verbose=False)

    banner = '=' * 60
    print(f"\n{banner}")
    print(f"  [FID] Epoch {epoch:4d}  VAE Reconstruction FID = {score:.4f}")
    print(f"{banner}\n")

    with open(fid_log_path, 'a') as f:
        f.write(json.dumps({'epoch': epoch, 'fid': round(score, 4)}) + '\n')

    return score






def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[VAE Training] device={device}")

    save_dir = Path(args.save_dir)
    viz_dir  = save_dir / 'reconstructions'
    fid_tmp  = save_dir / 'fid_tmp'
    fid_log  = save_dir / 'fid_log.jsonl'
    metrics_csv = save_dir / 'metrics.csv'
    metrics_fields = [
        'epoch',
        'train_loss', 'train_recon', 'train_kl', 'kl_weight',
        'train_mu_mean', 'train_mu_std', 'train_sigma_mean', 'train_sigma_std',
        'val_loss', 'val_recon', 'val_kl',
        'fid',
    ]
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    fid_tmp.mkdir(exist_ok=True)

    
    train_loader = build_dataloader(args.data_root, 'train', args.batch_size, args.workers)
    val_loader   = build_dataloader(args.data_root, 'val',   args.batch_size, args.workers)

    
    model = VAE().to(device)
    ema   = EMA(model, decay=0.999)
    log(f"[VAE] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    amp_enabled  = device.type == 'cuda'
    scaler       = make_scaler(amp_enabled)

    best_val_loss = float('inf')
    best_fid      = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()

        warmup_epochs = 10
        kl_weight = args.kl_weight * min(1.0, epoch / warmup_epochs)

        epoch_loss = epoch_recon = epoch_kl = 0.0
        mu_mean_sum = mu_std_sum = 0.0
        sigma_mean_sum = sigma_std_sum = 0.0
        mu_batches = 0
        n_batches  = len(train_loader)
        log(
            f"[Train] Epoch {epoch}/{args.epochs} starting: "
            f"{n_batches} batches, batch_size={args.batch_size}, workers={args.workers}"
        )
        log("[Train] Waiting for first batch from DataLoader...")

        for step, (imgs, _) in enumerate(train_loader):
            if step == 0:
                log("[Train] First batch loaded; GPU training has started.")
            imgs = imgs.to(device, non_blocking=True)

            with amp_context(device, amp_enabled):
                recon, mu, logvar = model(imgs)
                loss, recon_loss, kl_loss = vae_loss(recon, imgs, mu, logvar, kl_weight)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            epoch_loss  += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()

            mu_det = mu.detach()
            logvar_det = logvar.detach()
            sigma = (0.5 * logvar_det).exp()
            mu_mean_sum += mu_det.mean().item()
            mu_std_sum += mu_det.std(unbiased=False).item()
            sigma_mean_sum += sigma.mean().item()
            sigma_std_sum += sigma.std(unbiased=False).item()
            mu_batches += 1

            if step % args.log_every == 0:
                log(
                    f"Epoch {epoch:3d}/{args.epochs}  step {step:4d}/{n_batches}  "
                    f"loss={loss.item():.4f}  recon={recon_loss.item():.4f}  "
                    f"kl={kl_loss.item():.6f}  kl_w={kl_weight:.2e}"
                )

        lr_scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        train_recon = epoch_recon / max(n_batches, 1)
        train_kl = epoch_kl / max(n_batches, 1)
        if mu_batches > 0:
            train_mu_mean = mu_mean_sum / mu_batches
            train_mu_std = mu_std_sum / mu_batches
            train_sigma_mean = sigma_mean_sum / mu_batches
            train_sigma_std = sigma_std_sum / mu_batches
        else:
            train_mu_mean = train_mu_std = train_sigma_mean = train_sigma_std = 0.0

        
        backup = ema.backup()
        ema.apply_shadow()

        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, kl_weight, viz_dir, epoch
        )
        log(
            f"[Val] Epoch {epoch}  loss={val_loss:.4f}  "
            f"recon={val_recon:.4f}  kl={val_kl:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss},
                save_dir / 'vae_best.pt',
            )
            log(f"  ** New best VAE saved (val_loss={val_loss:.4f})")

        ema.restore(backup)

        
        if epoch % 10 == 0:
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()},
                save_dir / f'vae_epoch{epoch:04d}.pt',
            )

        
        fid_score = None
        if args.fid_every > 0 and epoch % args.fid_every == 0:
            bk = ema.backup()
            ema.apply_shadow()
            fid_score = compute_fid_vae(model, val_loader, device, fid_tmp, epoch, fid_log)
            if fid_score is not None and fid_score < best_fid:
                best_fid = fid_score
                torch.save(
                    {'epoch': epoch, 'model_state': model.state_dict(), 'fid': fid_score},
                    save_dir / 'vae_best_fid.pt',
                )
                log(f"  ** New best FID checkpoint saved (fid={fid_score:.4f})")
            ema.restore(bk)

        append_csv_row(
            metrics_csv,
            {
                'epoch': epoch,
                'train_loss': round(train_loss, 6),
                'train_recon': round(train_recon, 6),
                'train_kl': round(train_kl, 6),
                'kl_weight': float(kl_weight),
                'train_mu_mean': round(train_mu_mean, 6),
                'train_mu_std': round(train_mu_std, 6),
                'train_sigma_mean': round(train_sigma_mean, 6),
                'train_sigma_std': round(train_sigma_std, 6),
                'val_loss': round(val_loss, 6),
                'val_recon': round(val_recon, 6),
                'val_kl': round(val_kl, 6),
                'fid': '' if fid_score is None else round(fid_score, 6),
            },
            metrics_fields,
        )

    log(f"\n[VAE Training] Done.")
    log(f"  Best val_loss : {best_val_loss:.4f}")
    log(f"  Best FID      : {best_fid:.4f}")






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str,   required=True,
                   help='Path to Part A data directory')
    p.add_argument('--save_dir',   type=str,   default='checkpoints/vae')
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--kl_weight',  type=float, default=1e-6)
    p.add_argument('--workers',    type=int,   default=4)
    p.add_argument('--fid_every',  type=int,   default=10,
                   help='Compute reconstruction FID every N epochs. 0 = disable.')
    p.add_argument('--seed',       type=int,   default=775)
    p.add_argument('--log_every',  type=int,   default=100)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
