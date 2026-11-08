"""Differentiable Work1 adapter for Kimi-VL-A3B-Instruct."""

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from multimodalmodels.work1_token_adapter import (
    TokenReplacementWork1VLM,
    clip_normalized_to_pixels,
    normalize_pixels,
)


class KimiVLA3B(TokenReplacementWork1VLM):
    model_type = "kimi_vl"
    default_model_path = os.environ.get(
        "KIMIVL_MODEL_PATH", "moonshotai/Kimi-VL-A3B-Instruct"
    )

    def __init__(self, model_path=None, device=None, torch_dtype=torch.bfloat16,
                 image_size=448, max_txt_len=2048, system_prompt=None,
                 attn_implementation="sdpa"):
        model_path = str(model_path or self.default_model_path)
        device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.model_type != self.model_type:
            raise ValueError(
                f"Expected model_type={self.model_type}, got {config.model_type} at {model_path}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype,
            device_map={"": device}, attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        patch_size = int(config.vision_config.patch_size)
        merge_h, merge_w = map(int, config.vision_config.merge_kernel_size)
        image_size = int(image_size)
        if image_size % (patch_size * merge_h) or image_size % (patch_size * merge_w):
            raise ValueError(
                "Kimi-VL model_image_size must be divisible by patch_size*merge_kernel_size"
            )
        grid_h, grid_w = image_size // patch_size, image_size // patch_size
        image_token_estimate = grid_h * grid_w // (merge_h * merge_w)
        self.merge_kernel_size = (merge_h, merge_w)
        self._configure_work1(
            model=model, processor=processor, config=config, model_path=model_path,
            device=device, image_size=image_size, max_txt_len=max_txt_len,
            system_prompt=system_prompt, image_token=processor.image_token,
            image_token_id=config.media_placeholder_token_id,
            image_token_estimate=image_token_estimate, patch_size=patch_size,
        )
        self.merge_size = merge_h

    def _image_inputs(self, image, batch_size):
        pixels = normalize_pixels(
            clip_normalized_to_pixels(image), self.native_mean, self.native_std
        )
        _, channels, height, width = pixels.shape
        patch = self.patch_size
        if height != self.image_size or width != self.image_size:
            raise ValueError("Kimi-VL attack image has an unexpected resolution")
        grid_h, grid_w = height // patch, width // patch
        patches = (
            pixels.reshape(1, channels, grid_h, patch, grid_w, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(grid_h * grid_w, channels, patch, patch)
        )
        patches = (
            patches.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            .reshape(batch_size * grid_h * grid_w, channels, patch, patch)
        )
        patches = patches.to(dtype=self.model_dtype)
        grid = torch.tensor(
            [[grid_h, grid_w]], dtype=torch.long, device=image.device
        ).repeat(batch_size, 1)
        return {"pixel_values": patches, "image_grid_hws": grid}

    def _extract_image_features(self, image_inputs):
        return self.model._extract_image_features(
            image_inputs["pixel_values"], image_inputs["image_grid_hws"]
        )
