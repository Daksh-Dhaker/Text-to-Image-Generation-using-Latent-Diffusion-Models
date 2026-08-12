from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    def __init__(self, backbone: VisionTransformer, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for crop in crops:
            cls, _ = self.backbone(crop)
            outputs.append(self.head(cls))
        return outputs


def update_teacher_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int = 4096,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_outputs: List[torch.Tensor]) -> None:
        concat = torch.cat(teacher_outputs, dim=0)
        batch_center = concat.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        n_global_crops: int,
        n_local_crops: int,
    ) -> torch.Tensor:
        n_student_crops = n_global_crops + n_local_crops
        if len(student_outputs) < n_student_crops:
            raise ValueError(
                f"Expected at least {n_student_crops} student crops, got {len(student_outputs)}"
            )
        if len(teacher_outputs) < n_global_crops:
            raise ValueError(
                f"Expected at least {n_global_crops} teacher crops, got {len(teacher_outputs)}"
            )

        student_probs = [s / self.student_temp for s in student_outputs[:n_student_crops]]
        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_outputs[:n_global_crops]
        ]

        total_loss = 0.0
        n_terms = 0

        for iq, q in enumerate(teacher_probs):
            for v, s in enumerate(student_probs):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                total_loss += loss
                n_terms += 1

        total_loss = total_loss / max(n_terms, 1)
        self.update_center(teacher_outputs[:n_global_crops])
        return total_loss
