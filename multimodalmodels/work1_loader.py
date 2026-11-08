"""Resolve YAML model aliases, class names, or local model directories lazily."""

import importlib
import json
from pathlib import Path

import torch


ALIASES = {
    'qwen2_5_vl': 'qwen2_5_vl.Qwen25VL',
    'qwen3_vl': 'qwen3_vl.Qwen3VL',
    'glm4_1v': 'glm4_1v.GLM41V',
    'glm4v': 'glm4_1v.GLM41V',
    'glm-4.1v': 'glm4_1v.GLM41V',
    'glm-4.1v-9b-thinking': 'glm4_1v.GLM41V',
    'llava_v1_5_7b': 'llava.Llava15VL',
    'llava-1.5-7b': 'llava.Llava15VL',
    'llava_v1_6_vicuna': 'llava.Llava16Vicuna',
    'llava-v1.6-vicuna': 'llava.Llava16Vicuna',
    'llava-v1.6-vicuna-7b': 'llava.Llava16Vicuna',
    'kimi_vl_a3b': 'kimi_vl.KimiVLA3B',
    'kimi-vl-a3b': 'kimi_vl.KimiVLA3B',
}
ARCHITECTURES = {'qwen2_5_vl': ALIASES['qwen2_5_vl'],
                 'qwen3_vl': ALIASES['qwen3_vl'], 'glm4v': ALIASES['glm4v'],
                 'llava': ALIASES['llava_v1_5_7b'],
                 'llava_next': ALIASES['llava_v1_6_vicuna'],
                 'kimi_vl': ALIASES['kimi_vl_a3b']}


def resolve_work1_model(target_model, model_path=None):
    name = str(target_model or '')
    path = Path(name)
    if (path.is_absolute() or name.startswith(('./', '../'))) and not path.is_dir():
        raise FileNotFoundError(f'Local target_model directory does not exist: {path}')
    if path.is_dir():
        config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        try:
            resolved = ARCHITECTURES[config['model_type']]
        except KeyError as exc:
            raise ValueError(f'Unsupported local model_type: {config.get("model_type")}') from exc
        if model_path and Path(model_path).resolve() != path.resolve():
            raise ValueError('target_model directory conflicts with model_path')
        return resolved, str(path)
    name = ALIASES.get(name.lower(), name)
    if '.' not in name:
        raise ValueError('target_model must be a supported alias, local directory, or module.ClassName')
    return name, model_path or None


def load_work1_model(config):
    name, path = resolve_work1_model(config.target_model, getattr(config, 'model_path', None))
    module, class_name = name.rsplit('.', 1)
    model_class = getattr(importlib.import_module(f'multimodalmodels.{module}'), class_name)
    dtype_name = getattr(config, 'model_dtype', 'bfloat16')
    if dtype_name not in ('bfloat16', 'float16', 'float32'):
        raise ValueError(f'Unsupported model_dtype: {dtype_name}')
    common_kwargs = {
        'device': getattr(config, 'model_device', None),
        'torch_dtype': getattr(torch, dtype_name),
        'image_size': getattr(config, 'model_image_size', 448),
        'max_txt_len': getattr(config, 'model_max_txt_len', 2048),
        'attn_implementation': getattr(config, 'model_attn_implementation', 'sdpa'),
    }
    if path:
        common_kwargs['model_path'] = path
    if getattr(model_class, 'work1_native_backend', False):
        common_kwargs['system_prompt'] = getattr(config, 'model_system_prompt', None)
    return model_class(**common_kwargs)
