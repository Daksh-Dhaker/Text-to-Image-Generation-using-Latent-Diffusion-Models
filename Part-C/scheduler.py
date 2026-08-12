import math
from typing import Optional

import torch
import torch.nn.functional as F






def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps  = timesteps + 1
    t      = torch.linspace(0, timesteps, steps) / timesteps
    ab     = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab     = ab / ab[0]
    betas  = 1 - ab[1:] / ab[:-1]
    return betas.clamp(0.0, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)






class GaussianDiffusion(torch.nn.Module):


    def __init__(self, timesteps: int = 500, schedule: str = 'cosine'):
        super().__init__()
        self.timesteps = timesteps

        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Choose 'cosine' or 'linear'.")

        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        
        posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )

        self.register_buffer('betas',                          betas)
        self.register_buffer('alphas',                         alphas)
        self.register_buffer('alphas_cumprod',                 alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',            alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',            alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',  (1 - alphas_cumprod).sqrt())
        self.register_buffer('posterior_variance',             posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod),
        )

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(target_shape) - 1)))

    
    
    

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:

        sqrt_ab     = self._extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_ab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return sqrt_ab * z0 + sqrt_one_ab * noise

    
    def add_noise(self, z0, noise, t):
        return self.q_sample(z0, t, noise)

    
    
    

    def predict_z0_from_noise(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_one_ab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        return (zt - sqrt_one_ab * noise) / sqrt_ab

    def p_mean_variance(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        noise_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_z0 = self.predict_z0_from_noise(zt, t, noise_prediction).clamp(-5.0, 5.0)
        mean = (
            self._extract(self.posterior_mean_coef1, t, zt.shape) * pred_z0
            + self._extract(self.posterior_mean_coef2, t, zt.shape) * zt
        )
        log_variance = self._extract(self.posterior_log_variance_clipped, t, zt.shape)
        return mean, log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        zt: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        null_context: torch.Tensor,
        guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        eps_uncond = model(zt, t, null_context)
        eps_cond = model(zt, t, context)
        eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        mean, log_variance = self.p_mean_variance(zt, t, eps)
        noise = torch.randn_like(zt)
        nonzero_mask = (t != 0).float().view(zt.shape[0], *((1,) * (zt.dim() - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    
    
    

    def _ddim_step(
        self,
        eps: torch.Tensor,     
        t_cur: int,
        t_prev: int,
        zt: torch.Tensor,
        eta: float = 0.0,      
    ) -> torch.Tensor:
        ab_t    = self.alphas_cumprod[t_cur]
        ab_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 \
                  else torch.ones(1, device=zt.device)

        
        pred_z0 = (zt - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        pred_z0 = pred_z0.clamp(-5.0, 5.0)

        
        sigma  = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
        dir_zt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps

        z_prev = ab_prev.sqrt() * pred_z0 + dir_zt
        if eta > 0:
            z_prev = z_prev + sigma * torch.randn_like(zt)
        return z_prev

    
    
    

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        context: torch.Tensor,
        null_context: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 4.0,
        num_steps: Optional[int] = 50,
        eta: float = 0.0,
        sampler: str = 'ddim',
    ) -> torch.Tensor:

        zt = torch.randn(shape, device=device)

        sampler = sampler.lower()
        if sampler == 'ddpm':
            for t_int in range(self.timesteps - 1, -1, -1):
                t_batch = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
                zt = self.p_sample(model, zt, t_batch, context, null_context, guidance_scale=guidance_scale)
            return zt

        if sampler != 'ddim':
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddim' or 'ddpm'.")

        if num_steps is None:
            num_steps = self.timesteps
        num_steps = min(int(num_steps), self.timesteps)
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).round().long().tolist()

        for i, t_int in enumerate(timesteps):
            t_prev  = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)

            
            eps_uncond = model(zt, t_batch, null_context)
            eps_cond   = model(zt, t_batch, context)

            
            eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

            zt = self._ddim_step(eps, int(t_int), int(t_prev), zt, eta=eta)

        return zt   





DDPMScheduler = GaussianDiffusion
