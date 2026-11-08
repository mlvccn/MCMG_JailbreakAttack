"""Differentiable Work1 adapters for LLaVA-1.5 and LLaVA-NeXT Vicuna."""

import math
import os

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)
from transformers.image_processing_utils import select_best_resolution

from multimodalmodels.work1_token_adapter import (
    TokenReplacementWork1VLM,
    clip_normalized_to_pixels,
    normalize_pixels,
)


class _LlavaBase(TokenReplacementWork1VLM):
    model_class = None
    model_type = None
    default_model_path = None

    def __init__(self, model_path=None, device=None, torch_dtype=torch.float16,
                 image_size=336, max_txt_len=2048, system_prompt=None,
                 attn_implementation="sdpa"):
        model_path = str(model_path or self.default_model_path)
        device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type != self.model_type:
            raise ValueError(
                f"Expected model_type={self.model_type}, got {config.model_type} at {model_path}"
            )
        model = self.model_class.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map={"": device},
            attn_implementation=attn_implementation, low_cpu_mem_usage=True,
        )
        vision_size = int(config.vision_config.image_size)
        patch_size = int(config.vision_config.patch_size)
        image_token_estimate = self._visual_token_count(
            processor, config, int(image_size), vision_size, patch_size
        )
        self.vision_image_size = vision_size
        self._configure_work1(
            model=model, processor=processor, config=config, model_path=model_path,
            device=device, image_size=image_size, max_txt_len=max_txt_len,
            system_prompt=system_prompt, image_token=processor.image_token,
            image_token_id=config.image_token_index,
            image_token_estimate=image_token_estimate, patch_size=patch_size,
        )

    def _visual_token_count(self, processor, config, image_size, vision_size, patch_size):
        raise NotImplementedError

    def _extract_image_features(self, image_inputs):
        return self.model.get_image_features(**image_inputs)[0]


class Llava15VL(_LlavaBase):
    model_class = LlavaForConditionalGeneration
    model_type = "llava"
    default_model_path = os.environ.get(
        "LLAVA15_MODEL_PATH", "llava-hf/llava-1.5-7b-hf"
    )

    def _visual_token_count(self, processor, config, image_size, vision_size, patch_size):
        if image_size != vision_size:
            raise ValueError(
                f"LLaVA-1.5 requires model_image_size={vision_size}, got {image_size}"
            )
        default_count = (vision_size // patch_size) ** 2
        if config.vision_feature_select_strategy == "full":
            default_count += 1
        return int(getattr(config, "image_seq_length", default_count))

    def _image_inputs(self, image, batch_size):
        pixels = clip_normalized_to_pixels(image)
        if pixels.shape[-2:] != (self.vision_image_size, self.vision_image_size):
            raise ValueError("LLaVA-1.5 attack image has an unexpected resolution")
        pixels = normalize_pixels(
            pixels, self.native_mean, self.native_std
        ).to(dtype=self.model_dtype)
        return {"pixel_values": pixels.expand(batch_size, -1, -1, -1)}


class Llava16Vicuna(_LlavaBase):
    model_class = LlavaNextForConditionalGeneration
    model_type = "llava_next"
    default_model_path = os.environ.get(
        "LLAVA16_MODEL_PATH", "llava-hf/llava-v1.6-vicuna-7b-hf"
    )

    def _visual_token_count(self, processor, config, image_size, vision_size, patch_size):
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = patch_size
        # The default feature strategy removes CLIP's CLS token.
        processor.num_additional_image_tokens = (
            0 if config.vision_feature_select_strategy == "default" else 1
        )
        return int(processor._get_number_of_features(
            image_size, image_size, vision_size, vision_size
        ))

    def _anyres_patches(self, pixels):
        height, width = pixels.shape[-2:]
        target_h, target_w = select_best_resolution(
            [height, width], self.config.image_grid_pinpoints
        )
        scale_w, scale_h = target_w / width, target_h / height
        if scale_w < scale_h:
            new_w, new_h = target_w, min(math.ceil(height * scale_w), target_h)
        else:
            new_h, new_w = target_h, min(math.ceil(width * scale_h), target_w)
        resized = F.interpolate(
            pixels, size=(new_h, new_w), mode="bicubic", align_corners=False,
            antialias=True,
        )
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        padded = F.pad(resized, (left, right, top, bottom), value=0.0)
        crop = self.vision_image_size
        patches = (
            padded.unfold(2, crop, crop).unfold(3, crop, crop)
            .permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, crop, crop)
        )
        base = F.interpolate(
            pixels, size=(crop, crop), mode="bicubic", align_corners=False,
            antialias=True,
        )
        return torch.cat([base, patches], dim=0)

    def _image_inputs(self, image, batch_size):
        pixels = clip_normalized_to_pixels(image)
        height, width = pixels.shape[-2:]
        patches = normalize_pixels(
            self._anyres_patches(pixels), self.native_mean, self.native_std
        ).to(dtype=self.model_dtype)
        patches = patches.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        image_sizes = torch.tensor(
            [[height, width]], dtype=torch.long, device=image.device
        ).repeat(batch_size, 1)
        return {"pixel_values": patches, "image_sizes": image_sizes}
