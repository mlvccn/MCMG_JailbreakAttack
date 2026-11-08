"""Native single-image adapters for work1's model-independent optimization loop."""

from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import torch
from PIL import Image
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pack_image_patches(image, patch_size, temporal_patch_size, merge_size):
    """Match the HF processors' channel/time/merge ordering, retaining autograd."""
    batch, channels, height, width = image.shape
    factor = patch_size * merge_size
    if height % factor or width % factor:
        raise ValueError(f'Image dimensions must be divisible by patch_size*merge_size={factor}')
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image.unsqueeze(1).expand(-1, temporal_patch_size, -1, -1, -1)
    patches = patches.reshape(
        batch, 1, temporal_patch_size, channels,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(batch * grid_h * grid_w,
                           channels * temporal_patch_size * patch_size * patch_size)


class NativeWork1VLM:
    work1_native_backend = True
    hf_class_name = None
    default_model_path = None
    model_type = None

    def __init__(self, model_path=None, device=None, torch_dtype=torch.bfloat16,
                 image_size=448, max_txt_len=2048, system_prompt=None,
                 attn_implementation='sdpa'):
        import transformers
        from packaging.version import Version

        if not (Version('4.57.1') <= Version(transformers.__version__) < Version('5.0')):
            raise RuntimeError(
                f'{self.hf_class_name} requires transformers>=4.57.1,<5 for this adapter; '
                f'current version is {transformers.__version__}. '
                'Install transformers==4.57.3 in the training environment.'
            )
        model_class = getattr(transformers, self.hf_class_name, None)
        if model_class is None:
            raise RuntimeError(f'{self.hf_class_name} is unavailable in this Transformers installation')
        self.model_path = str(model_path or self.default_model_path)
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True,
        )
        config = transformers.AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if config.model_type != self.model_type:
            raise ValueError(f'Expected {self.model_type}, got {config.model_type} at {self.model_path}')
        self.configure_processor(self.processor, config, image_size, max_txt_len, system_prompt)
        self.model = model_class.from_pretrained(
            self.model_path, torch_dtype=torch_dtype, device_map={'': self.device},
            attn_implementation=attn_implementation, trust_remote_code=True,
        ).eval()
        self.language_model = self.model
        self._native_lock = RLock()

    def configure_processor(self, processor, config, image_size, max_txt_len=2048,
                            system_prompt=None):
        """Also usable with a tiny model in CPU contract tests."""
        self.processor, self.config = processor, config
        self.tokenizer = processor.tokenizer
        self.image_size, self.max_txt_len = int(image_size), int(max_txt_len)
        self.system_prompt = system_prompt
        vision = config.vision_config
        self.patch_size = int(vision.patch_size)
        self.temporal_patch_size = int(vision.temporal_patch_size)
        self.merge_size = int(vision.spatial_merge_size)
        if self.image_size < 1 or self.image_size % (self.patch_size * self.merge_size):
            raise ValueError('image_size must be a positive multiple of patch_size*spatial_merge_size')
        self.image_token = processor.image_token
        self.image_token_id = int(config.image_token_id)
        self.vision_start_token_id = int(
            config.image_start_token_id if self.model_type == 'glm4v' else config.vision_start_token_id
        )
        self.vision_end_token_id = int(
            config.image_end_token_id if self.model_type == 'glm4v' else config.vision_end_token_id
        )
        self.image_token_estimate = (self.image_size // self.patch_size // self.merge_size) ** 2
        image_processor = processor.image_processor
        self.native_mean = tuple(image_processor.image_mean)
        self.native_std = tuple(image_processor.image_std)
        self.vis_processor = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # work1's loop uses this tensor convention; native inputs below use
            # each model's own processor normalization (Qwen3 uses mean/std=0.5).
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        sentinel = '__WORK1_QUERY__'
        rendered = self.render_prompt(sentinel)
        image_marker = self.tokenizer.decode(
            [self.vision_start_token_id, self.image_token_id, self.vision_end_token_id],
            skip_special_tokens=False,
        )
        if rendered.count(image_marker) != 1 or rendered.count(sentinel) != 1:
            raise ValueError('Expected one image and one text slot in the native chat template')
        self.work1_prompt_template = (
            rendered.replace(image_marker, '<ImageHere>').replace('%', '%%').replace(sentinel, '%s')
        )

    def render_prompt(self, text):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
        if self.system_prompt is not None:
            messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def build_text_batch(self, queries, suffix, targets=None):
        if not queries or (targets is not None and len(queries) != len(targets)):
            raise ValueError('Nonempty queries and matching targets are required')
        ids, labels = [], []
        for index, query in enumerate(queries):
            prompt = self.render_prompt(query + suffix)
            prompt = prompt.replace(self.image_token, self.image_token * self.image_token_estimate)
            prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = [] if targets is None else self.tokenizer.encode(targets[index], add_special_tokens=False)
            if len(prefix_ids) + len(target_ids) > self.max_txt_len:
                raise ValueError('Native prompt + target exceeds model_max_txt_len; increase the YAML limit')
            ids.append(prefix_ids + target_ids)
            labels.append([-100] * len(prefix_ids) + target_ids)
        max_length = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError('Native tokenizer has neither pad_token_id nor eos_token_id')
        result = {
            'input_ids': torch.tensor([row + [pad] * (max_length - len(row)) for row in ids],
                                      dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor([[1] * len(row) + [0] * (max_length - len(row)) for row in ids],
                                           dtype=torch.long, device=self.device),
        }
        if targets is not None:
            result['labels'] = torch.tensor(
                [row + [-100] * (max_length - len(row)) for row in labels],
                dtype=torch.long, device=self.device,
            )
        counts = result['input_ids'].eq(self.image_token_id).sum(dim=1)
        if not torch.all(counts == self.image_token_estimate):
            raise ValueError('Native image placeholder count does not match the visual token grid')
        return result

    def prepare_pixel_values(self, clip_normalized_image, batch_size=1):
        image = clip_normalized_image.float()
        if image.shape != (1, 3, self.image_size, self.image_size):
            raise ValueError(f'Expected one RGB image at {self.image_size}x{self.image_size}')
        mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        pixels = image * std + mean
        native_mean = pixels.new_tensor(self.native_mean).view(1, 3, 1, 1)
        native_std = pixels.new_tensor(self.native_std).view(1, 3, 1, 1)
        pixels = (pixels - native_mean) / native_std
        packed = pack_image_patches(pixels, self.patch_size, self.temporal_patch_size, self.merge_size)
        grid = torch.tensor([[1, self.image_size // self.patch_size, self.image_size // self.patch_size]],
                            dtype=torch.long, device=image.device)
        return packed.repeat(batch_size, 1), grid.repeat(batch_size, 1)

    def build_visual_cache(self, image):
        pixels, grid = self.prepare_pixel_values(image)
        with self._native_lock, torch.no_grad():
            features = self.model.model.get_image_features(pixels, grid)
        if self.model_type == 'qwen3_vl':
            image_features, deepstack = features
            cache = {'image_features': image_features[0].detach(),
                     'deepstack': [value.detach() for value in deepstack]}
        else:
            cache = {'image_features': features[0].detach()}
        return {**cache, 'pixel_values': pixels.detach(), 'grid': grid.detach(),
                'model_type': self.model_type}

    @contextmanager
    def cached_image_features(self, cache, batch_size):
        """Let native forward perform RoPE/fusion, supplying all cached vision outputs."""
        core = self.model.model
        had_override = 'get_image_features' in vars(core)
        old_override = vars(core).get('get_image_features')

        def cached(pixel_values, image_grid_thw=None):
            images = tuple(cache['image_features'] for _ in range(batch_size))
            if self.model_type == 'qwen3_vl':
                return images, [value.repeat(batch_size, 1) for value in cache['deepstack']]
            return images

        core.get_image_features = cached
        try:
            yield
        finally:
            if had_override:
                core.get_image_features = old_override
            else:
                del core.get_image_features

    def forward_native(self, image, text_batch, visual_cache=None):
        batch_size = text_batch['input_ids'].shape[0]
        with self._native_lock:
            if visual_cache is None:
                pixels, grid = self.prepare_pixel_values(image, batch_size)
                return self.model(**text_batch, pixel_values=pixels, image_grid_thw=grid, use_cache=False)
            if torch.is_grad_enabled() and image.requires_grad:
                raise ValueError('Detached visual caches are for no-grad evaluation, not image-gradient updates')
            if visual_cache['model_type'] != self.model_type:
                raise ValueError('Visual cache belongs to a different model architecture')
            grid = visual_cache['grid'].repeat(batch_size, 1)
            pixels = visual_cache['pixel_values'].repeat(batch_size, 1)
            with self.cached_image_features(visual_cache, batch_size):
                return self.model(**text_batch, pixel_values=pixels, image_grid_thw=grid, use_cache=False)

    def generate_native(self, image, text, max_new_tokens=300):
        inputs = self.build_text_batch([text], '')
        pixels, grid = self.prepare_pixel_values(image)
        with self._native_lock, torch.no_grad():
            ids = self.model.generate(**inputs, pixel_values=pixels, image_grid_thw=grid,
                                      max_new_tokens=max_new_tokens, do_sample=False)
        generated = ids[0, inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def encode_img(self, image):
        raise RuntimeError('This adapter requires native forward; embedding-only fusion would omit model features')

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
                image = source.convert('RGB')
        tensor = self.vis_processor(image).unsqueeze(0).to(self.device)
        return self.generate_native(tensor, question, max_new_tokens)
