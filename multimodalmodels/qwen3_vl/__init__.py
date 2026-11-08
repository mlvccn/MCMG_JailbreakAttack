from multimodalmodels.work1_native import NativeWork1VLM


class Qwen3VL(NativeWork1VLM):
    hf_class_name = 'Qwen3VLForConditionalGeneration'
    default_model_path = 'Qwen/Qwen3-VL-8B-Instruct'
    model_type = 'qwen3_vl'
