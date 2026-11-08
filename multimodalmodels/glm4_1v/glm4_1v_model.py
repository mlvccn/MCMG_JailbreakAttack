"""GLM-4.1V adapter for the Work1 native multimodal optimization path."""

import os

from multimodalmodels.work1_native import NativeWork1VLM


DEFAULT_MODEL_PATH = os.environ.get(
    'GLM41V_MODEL_PATH',
    'zai-org/GLM-4.1V-9B-Thinking',
)


class GLM41V(NativeWork1VLM):
    """Expose GLM-4.1V through Work1's differentiable native-forward API."""

    hf_class_name = 'Glm4vForConditionalGeneration'
    default_model_path = DEFAULT_MODEL_PATH
    model_type = 'glm4v'

    def configure_processor(
        self,
        processor,
        config,
        image_size,
        max_txt_len=2048,
        system_prompt=None,
    ):
        super().configure_processor(
            processor,
            config,
            image_size,
            max_txt_len=max_txt_len,
            system_prompt=system_prompt,
        )
        required = ('image_start_token_id', 'image_end_token_id', 'image_token_id')
        missing = [name for name in required if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f'GLM-4.1V config is missing token IDs: {missing}')
        processor_image_id = getattr(processor, 'image_token_id', None)
        if processor_image_id is not None and int(processor_image_id) != self.image_token_id:
            raise ValueError(
                'GLM-4.1V processor image token does not match config.image_token_id'
            )
