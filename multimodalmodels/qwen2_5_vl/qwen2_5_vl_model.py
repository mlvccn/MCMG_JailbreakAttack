"""Qwen2.5-VL-7B model wrapper for white-box adversarial attack pipeline.

Provides the same interface expected by attacks/work1 (encode_img, tokenizer,
vis_processor, language_model, etc.) so that vlm_attacker / mcts / prompt_wrapper
can work without modification.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from multimodalmodels.work1_native import pack_image_patches


# Default model path — override via environment variable or constructor arg.
_DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN25VL_MODEL_PATH",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)


class Qwen25VL:
    """Thin wrapper around Qwen2.5-VL that exposes the interface required by
    the work1 adversarial attack pipeline.

    Key attributes expected downstream:
        - device
        - tokenizer          (for text tokenization)
        - language_model      (the causal LM backbone, for embed_tokens / forward)
        - vis_processor       (image to normalized tensor)
        - encode_img(tensor)  (differentiable image → hidden states)
        - max_txt_len
        - image_token_estimate  (number of visual tokens per image)
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        image_size: int = 448,
        max_txt_len: int = 2048,
        attn_implementation: str = "sdpa",
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ---------- load model & processor ----------
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": self.device},
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # ---------- expose tokenizer ----------
        self.tokenizer = self.processor.tokenizer

        # ---------- expose language model backbone ----------
        #  Qwen2_5_VLForConditionalGeneration
        #    .model           -> Qwen2_5_VLModel  (vision encoder + LM)
        #    .model.embed_tokens  -> nn.Embedding
        #    .lm_head         -> nn.Linear
        self.language_model = self.model
        self.llama_model = self.model          # alias for legacy code paths
        self.llama_tokenizer = self.tokenizer  # alias for legacy code paths

        self.max_txt_len = int(max_txt_len)

        # ---------- visual processor (PIL → normalised tensor) ----------
        # Matches Qwen2.5's CLIP normalization and work1's pixel convention.
        self.image_size = image_size
        self.vis_processor = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        # ---------- vision encoder for differentiable image embedding ----------
        self.visual = self.model.visual  # Qwen2_5_VisionTransformerPretrainedModel

        self.patch_size = int(getattr(self.visual, 'patch_size', 14))
        self.merge_size = int(getattr(self.visual, 'spatial_merge_size', 2))
        self.temporal_patch_size = int(getattr(self.visual, 'temporal_patch_size', 2))
        if self.image_size % (self.patch_size * self.merge_size):
            raise ValueError('image_size must be divisible by patch_size*spatial_merge_size')
        self.image_token_estimate = (self.image_size // self.patch_size // self.merge_size) ** 2

    # ------------------------------------------------------------------
    #  Core interface expected by prompt_wrapper / vlm_attacker / mcts
    # ------------------------------------------------------------------

    def prepare_pixel_values(self, img: torch.Tensor, batch_size=None):
        """Pack normalized RGB tensors exactly as the official image processor.

        Without batch_size, each input image keeps its own grid row. Passing
        batch_size broadcasts one shared image for work1's universal objective.
        No PIL/NumPy conversion is used, so image gradients remain connected.
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError('Expected normalized RGB images with shape [B, 3, H, W]')
        if batch_size is not None:
            if img.shape[0] != 1 or batch_size < 1:
                raise ValueError('Universal batching requires one shared image and a positive batch_size')
            img = img.expand(batch_size, -1, -1, -1)
        pixels = pack_image_patches(img, self.patch_size, self.temporal_patch_size, self.merge_size)
        height, width = img.shape[-2:]
        grid = torch.tensor(
            [[1, height // self.patch_size, width // self.patch_size]],
            device=img.device, dtype=torch.long,
        ).repeat(img.shape[0], 1)
        return pixels, grid

    def encode_img(self, img: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Differentiable image → visual hidden states.

        Args:
            img: normalised image tensor, shape (1, 3, H, W) or (B, 3, H, W)

        Returns:
            (image_embeds, None)  — second element retained by the Work1 interface
        """
        batch_size = img.shape[0]
        pixels, grid = self.prepare_pixel_values(img.to(device=self.device))
        image_embeds = self.visual(pixels.to(dtype=self.model.dtype), grid_thw=grid)
        
        # Reshape to (B, num_tokens, hidden_dim)
        num_tokens = image_embeds.shape[0] // batch_size
        image_embeds = image_embeds.reshape(batch_size, num_tokens, -1)

        return image_embeds, None

    # ------------------------------------------------------------------
    #  Delegate common methods to inner model
    # ------------------------------------------------------------------

    def eval(self):
        return self.model.eval()

    def zero_grad(self):
        return self.model.zero_grad()

    def requires_grad_(self, tag=False):
        return self.model.requires_grad_(tag)

    def parameters(self):
        return self.model.parameters()

    def to_type(self, dtype):
        return self.model.to(dtype)

    # ------------------------------------------------------------------
    #  High-level generation (for output preview during attack)
    # ------------------------------------------------------------------

    def query(self, img: Union[str, Image.Image], question: str) -> str:
        """Simple single-turn VQA (no gradient)."""
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            return "[qwen_vl_utils not installed, skipping preview]"

        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            return "[unsupported image type for preview]"

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": question},
            ]}
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def generate(self, test_cases, image_dir, **kwargs):
        outputs = []
        for image, prompt in test_cases:
            if isinstance(image, str):
                image = os.path.join(image_dir, image)
            output = self.query(image, prompt)
            outputs.append(output)
        return outputs
