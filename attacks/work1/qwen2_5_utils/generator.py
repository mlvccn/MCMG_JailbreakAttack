import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image


class Generator:
    """Qwen2.5-VL text generation helper for work1 attack pipeline.

    Supports two construction modes:
      1. Generator(model=<loaded_model>)     — reuse an already-loaded model (no extra GPU memory)
      2. Generator(model_path="Qwen/...")    — load a fresh model from path
    """

    def __init__(
        self,
        model=None,
        model_path: str = None,
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 300,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        if model is not None:
            # Reuse pre-loaded model (from Qwen25VL wrapper or raw HF model)
            # The wrapper exposes .model (the real HF model) and .tokenizer
            if hasattr(model, 'model') and hasattr(model, 'tokenizer'):
                self._hf_model = model.model  # inner HF model
                self._tokenizer = model.tokenizer
            else:
                self._hf_model = model
                self._tokenizer = getattr(model, 'tokenizer', None)
            self._device = next(self._hf_model.parameters()).device
        elif model_path is not None:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True, padding_side="left",
            )
            self._tokenizer = processor.tokenizer
            self._device = next(self._hf_model.parameters()).device
        else:
            raise ValueError("Generator requires either 'model' or 'model_path'")

    # ------------------------------------------------------------------
    #  Embedding-based generation (called from vlm_attacker with Prompt)
    # ------------------------------------------------------------------

    def generate(self, prompt_or_image, query: str = None, system_prompt: Optional[str] = None
                 ) -> Tuple[str, Any]:
        """Generate text. Supports two calling conventions:

        1. generate(prompt_obj)          — from a prompt_wrapper.Prompt (embedding-based preview)
        2. generate(image, query)        — from image path/PIL + text query
        """
        # Case 1: Prompt object (has context_embs)
        if hasattr(prompt_or_image, 'context_embs'):
            return self._generate_from_embeddings(prompt_or_image)

        # Case 2: image + query
        return self._generate_from_image(prompt_or_image, query, system_prompt)

    def _generate_from_embeddings(self, prompt) -> Tuple[str, List[int]]:
        """Generate from pre-computed context embeddings (for attack preview)."""
        context_embs = prompt.context_embs
        if not context_embs:
            return "[No context embeddings]", []

        # Use the first sample in the batch
        inputs_embeds = context_embs[0]  # (1, seq_len, hidden_dim)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )

        with torch.no_grad():
            outputs = self._hf_model.generate(
                inputs_embeds=inputs_embeds.to(self._hf_model.dtype),
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
            )

        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text, outputs[0].detach().cpu().tolist()

    def _generate_from_image(self, image, query, system_prompt=None) -> Tuple[str, List[int]]:
        """Generate from image + text query."""
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
        except ImportError as exc:
            return "[qwen_vl_utils not installed]", []

        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            return "[unsupported image type]", []

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": query},
        ]}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Need a processor for chat template — try to get from model or build one
        processor = getattr(self, '_processor', None)
        if processor is None:
            model_name = getattr(self._hf_model, 'name_or_path', None) or getattr(self._hf_model.config, '_name_or_path', '')
            if model_name:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
                self._processor = processor

        if processor is None:
            return "[Cannot build processor for image generation]", []

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._hf_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample, top_p=self.top_p,
                temperature=self.temperature, repetition_penalty=self.repetition_penalty,
            )
        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(trimmed[0], skip_special_tokens=True).strip()
        return text, trimmed[0].detach().cpu().tolist()

    def batch_generate(
        self, samples: Sequence[Tuple[Union[str, Image.Image], str]],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        outputs: List[str] = []
        for image, query in samples:
            text, _ = self.generate(image, query, system_prompt)
            outputs.append(text)
        return outputs
