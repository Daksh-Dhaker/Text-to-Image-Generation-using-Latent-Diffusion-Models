import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):


    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch, eps=1e-6)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()

        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        
        self.time_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None
            else None
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Encoder(nn.Module):


    def __init__(self):
        super().__init__()

        
        self.input_conv = nn.Conv2d(3, 32, 3, padding=1)

        
        self.stage1 = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32))
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  

        
        self.stage2 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  

        
        self.stage3 = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128))
        self.down3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  

        
        self.mid = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128))


        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)

        h = self.stage1(h)
        h = self.down1(h)

        h = self.stage2(h)
        h = self.down2(h)

        h = self.stage3(h)
        h = self.down3(h)

        h = self.mid(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h) 
        return h




class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        
        self.input_conv = nn.Conv2d(4, 128, 3, padding=1)

        
        self.mid = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128))

        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.stage1 = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128))

        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
        )
        self.stage2 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))

        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
        )
        self.stage3 = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32))

        
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=32, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z)
        h = self.mid(h)

        h = self.up1(h)
        h = self.stage1(h)

        h = self.up2(h)
        h = self.stage2(h)

        h = self.up3(h)
        h = self.stage3(h)

        h = self.act_out(self.norm_out(h))
        h = self.tanh(self.conv_out(h))
        return h






class VAE(nn.Module):


    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    
    def encode(self, x: torch.Tensor):

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        return self.decoder(z)

    def forward(self, x: torch.Tensor):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar




def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    recon_loss = F.mse_loss(recon, target, reduction='mean')


    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss
