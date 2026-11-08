"""
MGC: multi-view gradient coordination for VLM attacks.

Based on: "Boosting Adversarial Transferability through Augmentation in Hypothesis Space" (CVPR 2025)

Core idea: Average gradients over (perturbation neighbors x composed transform operators)
to produce more transferable adversarial images for Vision-Language Models.

Adapted for VLM: mild transforms only, small perturbation radius, baseline-weighted averaging.
Transforms are applied INSIDE the autograd graph so gradients are correctly mapped back.
"""

import random
import functools
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TFF


# ============================================================
# Image normalization (CLIP ViT, same as vlm_attacker.py)
# ============================================================

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


# ============================================================
# Basic Transform Operators  (input/output: NCHW tensor)
# All transforms are differentiable through PyTorch autograd.
# ============================================================

def vertical_shift(x):
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)


def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)


def vertical_flip(x):
    return x.flip(dims=(2,))


def horizontal_flip(x):
    return x.flip(dims=(3,))


def identity(x):
    return x


class scaling:
    """Scale pixel values by a constant factor."""
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x / self.scale


class rotate:
    """Rotate image by a fixed angle."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TFF.rotate(img=x, angle=self.angle)


class dim:
    """Diverse Input Method - random resize + pad + resize back."""
    def __init__(self, resize_rate=1.1):
        self.resize_rate = resize_rate

    def __call__(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        rnd = torch.randint(
            low=min(img_size, img_resize),
            high=max(img_size, img_resize),
            size=(1,), dtype=torch.int32,
        )
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)


# ============================================================
# VLM-adapted OPS Enhancer
# ============================================================

class VLMOPSEnhancer:
    """
    OPS-style averaged gradient for VLM adversarial attacks.

    Key VLM adaptations vs. original OPS:
      - Only mild, semantics-preserving transforms (no large rotations/scaling).
      - Small perturbation neighbourhood radius (beta=0.3 vs 2.0).
      - Baseline-weighted gradient averaging (baseline_weight=0.6).
      - Low sampling counts (2 neighbours x 3 operators = 6 extra forwards).
      - Transforms applied INSIDE autograd graph for correct gradient mapping.

    The compute_grad_fn callback must accept an optional `transform` kwarg:
        def compute_grad_fn(img_01, transform=None):
            img_01 = img_01.detach().clone().requires_grad_(True)
            img = transform(img_01) if transform is not None else img_01
            x_adv = normalize(img)
            ...
            grad = torch.autograd.grad(loss, img_01)[0]
            return grad
    """

    def __init__(
        self,
        epsilon=16 / 255,
        beta=0.3,
        num_sample_neighbor=2,
        num_sample_operator=3,
        sample_levels=None,
        sample_ratios=None,
        baseline_weight=0.6,
        device='cuda:0',
    ):
        self.epsilon = epsilon
        self.device = device
        self.num_sample_neighbor = num_sample_neighbor
        self.num_sample_operator = num_sample_operator
        self.baseline_weight = baseline_weight

        if sample_levels is None:
            sample_levels = [2]
        self.sample_levels = sample_levels

        if sample_ratios is None:
            sample_ratios = np.array([0.25, 0.5, 0.75])
        self.sample_radius = beta * epsilon * sample_ratios

        # Only mild, semantics-preserving transforms for VLM compatibility
        self.basic_ops = [
            identity,
            vertical_flip, horizontal_flip,
            vertical_shift, horizontal_shift,
            rotate(5), rotate(-5), rotate(10), rotate(-10),
            dim(1.05), dim(1.1), dim(1.15),
        ]

        self._op_list = []
        self._eps_list = []

    # ---- Operator Sampling ----

    def _get_new_ops(self, k=2):
        """Compose k randomly chosen basic operators into one transform."""
        sel_ops = random.choices(self.basic_ops, k=k)
        return functools.reduce(
            lambda f, g: lambda x, _f=f, _g=g: _f(_g(x)),
            sel_ops,
            identity,
        )

    def _init_op_list(self):
        self._op_list = []
        for level in self.sample_levels:
            if level == 1:
                self._op_list.extend(list(self.basic_ops))
            else:
                for _ in range(len(self.basic_ops)):
                    self._op_list.append(self._get_new_ops(k=level))

    # ---- Perturbation Sampling ----

    def _init_eps_list(self, shape):
        self._eps_list = []
        for radius in self.sample_radius:
            for _ in range(self.num_sample_neighbor):
                noise = torch.zeros(shape).uniform_(-radius, radius).to(self.device)
                self._eps_list.append(noise)

    # ---- Core: Averaged Gradient ----

    def get_averaged_gradient(self, adv_noise, compute_grad_fn):
        """
        Compute the OPS-averaged gradient with baseline-weighted averaging.

        The transform is passed INTO compute_grad_fn so that autograd's chain rule
        correctly maps gradients from the transformed space back to the original space.

        Args:
            adv_noise: image tensor (N, C, H, W) in [0, 1].
            compute_grad_fn: callable (img_01, transform=None) -> gradient tensor.
                Must apply transform INSIDE the autograd graph before the model forward.

        Returns:
            weighted gradient: baseline_weight * baseline + (1-baseline_weight) * mean(augmented)
        """
        baseline_grad = compute_grad_fn(adv_noise)

        self._init_op_list()
        self._init_eps_list(adv_noise.shape)

        selected_eps = random.sample(
            self._eps_list,
            min(self.num_sample_neighbor, len(self._eps_list)),
        )

        aug_grad_sum = torch.zeros_like(baseline_grad)
        aug_count = 0
        for eps in selected_eps:
            x_near = (adv_noise.detach() + eps).clamp(0, 1)

            selected_ops = random.sample(
                self._op_list,
                min(self.num_sample_operator, len(self._op_list)),
            )

            for op in selected_ops:
                grad = compute_grad_fn(x_near, transform=op)
                aug_grad_sum = aug_grad_sum + grad
                aug_count += 1

        if aug_count > 0:
            avg_aug_grad = aug_grad_sum / aug_count
            return self.baseline_weight * baseline_grad + (1 - self.baseline_weight) * avg_aug_grad
        return baseline_grad

    def get_averaged_gradient_efficient(
        self, adv_noise, compute_grad_fn, return_stability: bool = False,
        diagnostics=None,
    ):
        """
        Memory-efficient variant: clears GPU cache after each neighbour batch.
        Recommended for large VLMs (>=7B).

        When ``return_stability`` is enabled, the method also computes a
        per-pixel direction-consistency map from the same augmented gradients
        that form ``avg_grad``. This avoids a second, independently sampled set
        of backward passes and keeps the mask aligned with the update gradient.
        """
        baseline_grad = compute_grad_fn(adv_noise)

        self._init_op_list()
        self._init_eps_list(adv_noise.shape)

        selected_eps = random.sample(
            self._eps_list,
            min(self.num_sample_neighbor, len(self._eps_list)),
        )

        aug_grad_sum = torch.zeros_like(baseline_grad)
        unit_grad_sum = (
            torch.zeros_like(baseline_grad, dtype=torch.float32)
            if return_stability else None
        )
        direction_count = (
            torch.zeros_like(baseline_grad[:, :1], dtype=torch.int16)
            if diagnostics is not None and return_stability else None
        )
        aug_count = 0
        for eps in selected_eps:
            x_near = (adv_noise.detach() + eps).clamp(0, 1)

            selected_ops = random.sample(
                self._op_list,
                min(self.num_sample_operator, len(self._op_list)),
            )

            for op in selected_ops:
                grad = compute_grad_fn(x_near, transform=op)
                aug_grad_sum = aug_grad_sum + grad
                if unit_grad_sum is not None:
                    grad_float = grad.detach().float()
                    unit_grad_sum.add_(
                        F.normalize(grad_float, dim=1)
                    )
                    if direction_count is not None:
                        norms = torch.linalg.vector_norm(grad_float, dim=1, keepdim=True)
                        direction_count.add_((torch.isfinite(norms) & (norms > 1e-12)).short())
                aug_count += 1

                del grad
            gc.collect()
            torch.cuda.empty_cache()

        if aug_count > 0:
            avg_aug_grad = aug_grad_sum / aug_count
            # return self.baseline_weight * baseline_grad + (1 - self.baseline_weight) * avg_aug_grad
            if not return_stability:
                return avg_aug_grad

            if aug_count < 2:
                stability = torch.zeros_like(avg_aug_grad[:, :1], dtype=torch.float32)
            else:
                avg_direction = F.normalize(avg_aug_grad.detach().float(), dim=1)
                mean_unit_direction = unit_grad_sum / aug_count
                stability = (mean_unit_direction * avg_direction).sum(
                    dim=1, keepdim=True
                )
            if diagnostics is not None:
                avg_norm = torch.linalg.vector_norm(
                    avg_aug_grad.detach().float(), dim=1, keepdim=True
                )
                diagnostics.update({
                    'stability': stability.detach(),
                    'valid_mask': (
                        (direction_count >= 2) & torch.isfinite(stability)
                        & torch.isfinite(avg_norm) & (avg_norm > 1e-12)
                    ),
                    'valid_gradient_samples': direction_count,
                    'aug_count': aug_count,
                    'direction_norm_epsilon': 1e-12,
                })
            return avg_aug_grad, stability

        if return_stability:
            # A single baseline gradient contains no cross-sample consistency
            # information, so use a neutral mask rather than amplifying it.
            stability = torch.zeros_like(baseline_grad[:, :1], dtype=torch.float32)
            if diagnostics is not None:
                diagnostics.update({
                    'stability': stability,
                    'valid_mask': torch.zeros_like(stability, dtype=torch.bool),
                    'valid_gradient_samples': direction_count,
                    'aug_count': 0,
                    'direction_norm_epsilon': 1e-12,
                })
            return baseline_grad, stability
        return baseline_grad

    # ================================================================
    #  Cross-Modal Coordination Reweighting (Phase 1)
    #  通过梯度方向稳定性度量图文扰动协同程度，重加权像素梯度
    # ================================================================

    def get_averaged_gradient_with_coordination(
        self, adv_noise, compute_grad_fn,
        coordination_strength: float = 0.3,
        num_coordination_samples: int = 3,
        return_soft_mask: bool = False,
        return_diagnostics: bool = False,
    ):
        """OPS 平均梯度 + 跨模态协同重加权。

        Args:
            adv_noise: 对抗图像 (1, C, H, W) in [0, 1]
            compute_grad_fn: 梯度计算回调 (img_01, transform=None) -> grad
            coordination_strength: 协同重加权强度 (0=不使用, 0.3=轻度, 0.5=中度)
            num_coordination_samples: 兼容旧配置；小于 2 时关闭一致性重加权。
                启用后复用所有参与 OPS 平均的梯度，不再额外反向传播。
            return_soft_mask: 若为 True，返回 (avg_grad, soft_mask)，由调用方在
                sign(avg_grad) 之后施加 soft_mask，以保留连续空间权重
            return_diagnostics: Return (avg_grad, actual_weight, diagnostics).
                Reuses the same gradients even when coordination is disabled;
                the actual weight remains all ones in that case.

        Returns:
            默认返回重加权梯度；return_soft_mask=True 时返回
            (avg_grad, soft_mask)，其中 soft_mask 为连续正值空间权重
        """
        diagnostics = {} if return_diagnostics else None
        if coordination_strength <= 0 and not return_diagnostics:
            avg_grad = self.get_averaged_gradient_efficient(
                adv_noise, compute_grad_fn
            )
            if return_soft_mask:
                soft_mask = torch.ones_like(avg_grad[:, :1])
                return avg_grad, soft_mask
            return avg_grad
        avg_grad, stability = self.get_averaged_gradient_efficient(
            adv_noise, compute_grad_fn, return_stability=True,
            diagnostics=diagnostics,
        )

        if coordination_strength <= 0:
            W = torch.ones_like(avg_grad[:, :1])
        else:
            stability_01 = stability.clamp(min=0.0, max=1.0)
            centered_stability = 2.0 * stability_01 - 1.0
            W = 1.0 + coordination_strength * centered_stability
            W = W.clamp(min=0.4, max=2.0)  # [1, 1, H, W]

        if return_diagnostics:
            return avg_grad, W, diagnostics

        # 符号梯度更新必须在 sign() 之后应用 W，否则正值 W 会被 sign 抵消。
        if return_soft_mask:
            return avg_grad, W

        # Step 5: 应用权重
        coordinated_grad = avg_grad * W               # [C, H, W] * [1, H, W]

        gc.collect()
        torch.cuda.empty_cache()

        return coordinated_grad
