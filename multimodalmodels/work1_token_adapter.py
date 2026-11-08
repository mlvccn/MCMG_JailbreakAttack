"""Shared Work1 adapter for VLMs that replace image placeholder tokens."""

from pathlib import Path
from threading import RLock

import torch
from PIL import Image
from torchvision import transforms

from multimodalmodels.work1_native import CLIP_MEAN, CLIP_STD


class TokenReplacementWork1VLM:
    """Native-forward contract for LLaVA-style image token replacement."""

    work1_native_backend = True

    def _configure_work1(
        self,
        *,
        model,
        processor,
        config,
        model_path,
        device,
        image_size,
        max_txt_len,
        system_prompt,
        image_token,
        image_token_id,
        image_token_estimate,
        patch_size,
    ):
        self.model = model.eval()
        self.model_dtype = next(model.parameters()).dtype
        self.processor = processor
        self.config = config
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.max_txt_len = int(max_txt_len)
        self.system_prompt = system_prompt
        self.tokenizer = processor.tokenizer
        self.language_model = model
        self.image_token = image_token
        self.image_token_id = int(image_token_id)
        self.image_token_estimate = int(image_token_estimate)
        self.patch_size = int(patch_size)
        self.merge_size = 1
        self.vision_start_token_id = self.image_token_id
        self.vision_end_token_id = self.image_token_id
        image_processor = processor.image_processor
        self.native_mean = tuple(image_processor.image_mean)
        self.native_std = tuple(image_processor.image_std)
        self.vis_processor = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self._native_lock = RLock()

        sentinel = "__WORK1_QUERY__"
        rendered = self.render_prompt(sentinel)
        if rendered.count(self.image_token) != 1 or rendered.count(sentinel) != 1:
            raise ValueError("Expected one image placeholder and one text slot in the chat template")
        self.work1_prompt_template = (
            rendered.replace(self.image_token, "<ImageHere>")
            .replace("%", "%%")
            .replace(sentinel, "%s")
        )

    def render_prompt(self, text):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text},
        ]}]
        if self.system_prompt is not None:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def build_text_batch(self, queries, suffix, targets=None):
        if not queries or (targets is not None and len(queries) != len(targets)):
            raise ValueError("Nonempty queries and matching targets are required")
        ids, labels = [], []
        expanded_image = self.image_token * self.image_token_estimate
        for index, query in enumerate(queries):
            prompt = self.render_prompt(query + suffix)
            if prompt.count(self.image_token) != 1:
                raise ValueError("Chat template did not render exactly one image placeholder")
            prompt = prompt.replace(self.image_token, expanded_image)
            prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = [] if targets is None else self.tokenizer.encode(
                targets[index], add_special_tokens=False
            )
            if len(prefix_ids) + len(target_ids) > self.max_txt_len:
                raise ValueError("Prompt + target exceeds model_max_txt_len")
            ids.append(prefix_ids + target_ids)
            labels.append([-100] * len(prefix_ids) + target_ids)
        max_length = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        batch = {
            "input_ids": torch.tensor(
                [row + [pad] * (max_length - len(row)) for row in ids],
                dtype=torch.long, device=self.device,
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (max_length - len(row)) for row in ids],
                dtype=torch.long, device=self.device,
            ),
        }
        if targets is not None:
            batch["labels"] = torch.tensor(
                [row + [-100] * (max_length - len(row)) for row in labels],
                dtype=torch.long, device=self.device,
            )
        counts = batch["input_ids"].eq(self.image_token_id).sum(dim=1)
        if not torch.all(counts == self.image_token_estimate):
            raise ValueError("Image placeholder count does not match visual feature count")
        return batch

    def _image_inputs(self, image, batch_size):
        raise NotImplementedError

    def _extract_image_features(self, image_inputs):
        raise NotImplementedError

    def build_visual_cache(self, image):
        image_inputs = self._image_inputs(image, 1)
        with self._native_lock, torch.no_grad():
            features = self._extract_image_features(image_inputs)
        if features.shape[0] != self.image_token_estimate:
            raise ValueError("Vision encoder output does not match image_token_estimate")
        return {"image_features": features.detach(), "model_type": self.model_type}

    def _forward_with_cached_image(self, text_batch, cache):
        if cache["model_type"] != self.model_type:
            raise ValueError("Visual cache belongs to a different model architecture")
        input_ids = text_batch["input_ids"]
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        features = cache["image_features"].to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        mask = input_ids.eq(self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if mask.sum().item() != features.numel():
            raise ValueError("Cached visual features do not match image placeholders")
        inputs_embeds = inputs_embeds.masked_scatter(mask, features)
        kwargs = {key: value for key, value in text_batch.items() if key != "input_ids"}
        return self.model(inputs_embeds=inputs_embeds, use_cache=False, **kwargs)

    def forward_native(self, image, text_batch, visual_cache=None):
        batch_size = text_batch["input_ids"].shape[0]
        with self._native_lock:
            if visual_cache is not None:
                if torch.is_grad_enabled() and image.requires_grad:
                    raise ValueError("Detached visual caches cannot be used for image-gradient updates")
                return self._forward_with_cached_image(text_batch, visual_cache)
            return self.model(
                **text_batch, **self._image_inputs(image, batch_size), use_cache=False
            )

    def generate_native(self, image, text, max_new_tokens=300):
        inputs = self.build_text_batch([text], "")
        with self._native_lock, torch.no_grad():
            ids = self.model.generate(
                **inputs, **self._image_inputs(image, 1),
                max_new_tokens=max_new_tokens, do_sample=False,
            )
        generated = ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def encode_img(self, image):
        raise RuntimeError("This adapter requires native multimodal forward")

    def eval(self):
        return self.model.eval()

    def requires_grad_(self, value=False):
        return self.model.requires_grad_(value)

    def zero_grad(self):
        return self.model.zero_grad()

    def parameters(self):
        return self.model.parameters()

    def query(self, image, question, max_new_tokens=512):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        tensor = self.vis_processor(image).unsqueeze(0).to(self.device)
        return self.generate_native(tensor, question, max_new_tokens)


def clip_normalized_to_pixels(image):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected one CLIP-normalized RGB image")
    mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return image.float() * std + mean


def normalize_pixels(image, mean, std):
    mean = image.new_tensor(mean).view(1, 3, 1, 1)
    std = image.new_tensor(std).view(1, 3, 1, 1)
    return (image - mean) / std
