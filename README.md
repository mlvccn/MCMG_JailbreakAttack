# README

## Official code of "MCMG: Multimodal Jailbreak Attacks via Monte-Carlo Tree Context Search and Multi-view Gradient Consistency"

[![Python](https://img.shields.io/badge/Python-3.10.19-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-blue)](https://pytorch.org/) [![CUDA](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)


## Introduction

Jailbreak attacks act as an important way of evaluating the safety alignment and robustness of multimodal large language models. Existing universal multimodal jailbreak attacks face two major problems: 1) When text suffix search selects candidate text suffixes via gradient-estimated loss, due to text discreteness, the estimated values are often inconsistent with the loss values of model, and the sequential relationships among tokens are not fully considered; 2) Since visual perturbations rely on the current training text query and spatial gradients of image, the attack ability on the current training query is difficult to generalize to other text quries. To address the above problems, this paper proposes a multimodal jailbreak attack method via Monte-Carlo Tree Context Search and Multi-view Gradient Consistency (MCMG). The former Monte-carlo tree Context Search (MCS) module  seeks text suffix by context modeling of the latent token sequential relations, while the latter Multi-view Gradient Consistency (MGC) module generates perturbation by dynamic modeling of multi-view pixel-wise spatial relation.  
In particular, the MCS module makes candiate token suffixes by employing their embedding similarity and transfer probability as priori, and models the text suffix search context by Monte-Carlo tree. This yields a set of complete candiate text suffixes, which are selected in a coarse-to-fine manner on training queries and validation ones respectively. The selected candiate suffix replaces the original suffix in a tree, when it degrades the loss of model. Moreover, the MGC module obtains multi-view images by pixel-level neighborhood sampling of the current image with combination transformations, and computes their gradients. By imposing pixel-level gradient consistency among multi-view images, we obtain soft masks to dynamically adjust step for generating stronger perturbations. By freezing the model parameters, text suffix and adversarial image are updated alternatively, using the weighted cross-entropy loss of prediction probabilities for both modules. 
We generate jailbreak samples on AdvBench dataset to examine our attacker under the settings of white-box/cross-modal attacks, and denfense, on two benchmarks including HarmBench and MM-SafetyBench. Results demonstrate the stronger attack abilities under different text queries and target modals.  

![](figs/framework.png)

## Table of contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Attack and Reference](#attack-and-reference)
- [Quantitative Results](#quantitative-results)
- [Qualitative Results](#qualitative-results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [License](#license)


## Installation

The code was tested on Ubuntu 20.04.6 LTS. Training and vLLM inference use separate environments because the tested training stack uses PyTorch 2.6.0 with CUDA 11.8, while the tested vLLM stack uses PyTorch 2.8.0 with CUDA 12.8.

Install [Conda](https://docs.conda.io/projects/miniconda/en/latest/) and create the training environment from the repository root:

```bash
conda env create -f environment.yml
conda activate work1-train
pip install -r requirements-train.txt
pip install -e . --no-deps
```

Create the vLLM environment used for response generation and HarmBench evaluation:

```bash
conda env create -f environment-vllm.yml
conda activate work1-vllm
pip install -r requirements-inference.txt
```

## Dataset

The release uses one optimization corpus and one independent evaluation set:

- `attacks/data/harmful_behaviors.csv`: optimization query-target pairs. A fixed subset of 66 examples is sampled with `random_seed: 42`.
- `dataset/harmbench_stand200.json`: 200 independent evaluation behaviors.
- `attacks/data/clean.jpeg`: initial image used to construct the universal image perturbation.
- `attacks/data/qwen_bigram_logprobs.json`: Qwen tokenizer bigram prior used by suffix search.

The expected layout is:

```text
├── attacks
│   └── data
│       ├── clean.jpeg
│       ├── harmful_behaviors.csv
│       └── qwen_bigram_logprobs.json
└── dataset
    ├── apple.png
    └── harmbench_stand200.json
```

Each object in `dataset/harmbench_stand200.json` follows this format:

```json
{
  "image_path": "dataset/apple.png",
  "original_prompt": "<behavior>",
  "style": "declarative",
  "main_category": "standard",
  "subcategory": "<category>",
  "id": "<unique_id>"
}
```


## Models

Download the target model and HarmBench classifier, then set their local paths in the commands and configuration files below.

```text
├── MllmWeights
│   ├── Qwen2.5-VL-7B-Instruct
│   ├── Qwen3-VL-8B-Instruct
│   ├── GLM-4.1V-9B-Thinking
│   ├── llava-1.5-7b-hf
│   ├── llava-v1.6-vicuna-7b-hf
│   ├── Kimi-VL-A3B-Instruct
│   └── HarmBench-Llama-2-13b-cls
```

### Qwen2.5-VL-7B-Instruct

Set `target_model` in `configs/work1_qwen2_5_vl.yaml` to the model directory:

```yaml
target_model: "/path/to/Qwen2.5-VL-7B-Instruct"
image_gradient_sub_batch_size: 2
```

The complete reference configuration is provided in `configs/work1_qwen2_5_vl.yaml`.

### Qwen3-VL-8B-Instruct

```yaml
target_model: "/path/to/Qwen3-VL-8B-Instruct"
image_gradient_sub_batch_size: 1
```

The complete reference configuration is provided in `configs/work1_qwen3_vl.yaml`.

<!-- ### GLM-4.1V-9B-Thinking

```yaml
target_model: "/path/to/GLM-4.1V-9B-Thinking"
image_gradient_sub_batch_size: 1
```

The complete reference configuration is provided in `configs/work1_glm4_1v.yaml`.

### LLaVA-1.5-7B

```yaml
target_model: "/path/to/llava-1.5-7b-hf"
model_dtype: "float16"
model_image_size: 336
```

The complete configuration is provided in `configs/work1_llava_v1_5_7b.yaml`.

### LLaVA-1.6-Vicuna-7B

```yaml
target_model: "/path/to/llava-v1.6-vicuna-7b-hf"
model_dtype: "float16"
model_image_size: 336
```

The complete configuration is provided in `configs/work1_llava_v1_6_vicuna.yaml`.

### Kimi-VL-A3B-Instruct

```yaml
target_model: "/path/to/Kimi-VL-A3B-Instruct"
model_dtype: "bfloat16"
model_image_size: 448 -->
<!-- ```

The complete configuration is provided in `configs/work1_kimi_vl_a3b.yaml`. -->

### HarmBench-Llama-2-13b-cls

The evaluator reads the behavior from `original_prompt`, the generation from `model_response`, and reports ASR as the fraction classified as `yes`.


## Attack and Reference

All commands in this section are run from the repository root.

### 1. Configure the target model

This is a prerequisite configuration step, not a separate runtime command. Choose
one configuration file and set its local `target_model` path before running the
commands in Section 2:

| Configuration | `target_model` |
|---|---|
| `configs/work1_qwen2_5_vl.yaml` | `/path/to/Qwen2.5-VL-7B-Instruct` |
| `configs/work1_qwen3_vl.yaml` | `/path/to/Qwen3-VL-8B-Instruct` |
| `configs/work1_glm4_1v.yaml` | `/path/to/GLM-4.1V-9B-Thinking` |
| `configs/work1_llava_v1_5_7b.yaml` | `/path/to/llava-1.5-7b-hf` |
| `configs/work1_llava_v1_6_vicuna.yaml` | `/path/to/llava-v1.6-vicuna-7b-hf` |
| `configs/work1_kimi_vl_a3b.yaml` | `/path/to/Kimi-VL-A3B-Instruct` |

The remaining paths, including `prior_image`, `corpus`, and `save_dir`, are
already defined in the reference configurations. For Qwen2.5-VL, set
`bigram_tokenizer_path` to the same local Qwen2.5-VL directory when the bigram
prior is enabled.

### 2. Generate the universal adversarial image and suffix

After completing the configuration step above, directly run the command for the
selected model. Activate `work1-train` first.

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_qwen2_5_vl.yaml \
  --behaviors-file dataset/harmbench_stand200.json
```

<!-- For GLM-4.1V, select its configuration and keep the image-gradient sub-batch at
one unless the available memory permits a larger value:

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_glm4_1v.yaml \
  --behaviors-file dataset/harmbench_stand200.json
``` -->

For Qwen3-VL, use its model-specific configuration in the same way:

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_qwen3_vl.yaml \
  --behaviors-file dataset/harmbench_stand200.json
```

<!-- Run LLaVA-1.5-7B with its model-specific configuration:

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_llava_v1_5_7b.yaml \
  --behaviors-file dataset/harmbench_stand200.json
```

Run LLaVA-1.6-Vicuna-7B:

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_llava_v1_6_vicuna.yaml \
  --behaviors-file dataset/harmbench_stand200.json
```

Run Kimi-VL-A3B-Instruct:

```bash
conda activate work1-train
CUDA_VISIBLE_DEVICES=0 python run_work1.py \
  --config configs/work1_kimi_vl_a3b.yaml \
  --behaviors-file dataset/harmbench_stand200.json
``` -->

The reference configuration optimizes a fixed 66-example subset for 2,000 iterations and writes the following artifacts to `save_dir`:

```text
├── audit/<run_id>/
├── checkpoint_iter_500.pt
├── checkpoint_iter_1000.pt
├── checkpoint_iter_1500.pt
├── checkpoint_iter_2000.pt
├── effective_config.yaml
├── loss_curve.png
├── selected_training_indices.json
├── test_cases.jsonl
├── work1_universal.png
└── work1_universal_iter_*.png
```

<!-- Each checkpoint stores the adversarial suffix, image tensor, iteration number, losses, and an attack-configuration snapshot. The image perturbation is projected to the configured `eps / 255` L-infinity neighborhood around the initial image. -->

### 3. Export every checkpoint as an evaluation set

```bash
conda activate work1-train
python generate_work1_checkpoint_cases.py \
  output/test_cases/work1_qwen2_5_vl
```

This creates `test_cases_iter_<iteration>.jsonl` and the corresponding PNG image for every checkpoint.
Export the additional model checkpoints in the same way.

<!-- ```bash
conda activate work1-train
python generate_work1_checkpoint_cases.py \
  output/test_cases/work1_llava_v1_5_7b
python generate_work1_checkpoint_cases.py \
  output/test_cases/work1_llava_v1_6_vicuna
python generate_work1_checkpoint_cases.py \
  output/test_cases/work1_kimi_vl_a3b
``` -->

### 4. Generate model responses with vLLM

Activate `work1-vllm` and run:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 ./run_generate_qwen2_5_checkpoints.sh \
  output/test_cases/work1_qwen2_5_vl \
  --model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --model_name qwen2_5_vl \
  --batch_size 32 \
  --max_tokens 512 \
  --max_model_len 8192 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
```

The generated files are named `test_generate_cases_iter_<iteration>.jsonl`. Each file has an adjacent `*.generation_config.json` record containing the input checksum, model arguments, sampling parameters, chat template, and package versions.

For Qwen3-VL checkpoints:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 QWEN3_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
  ./run_generate_qwen3_checkpoints.sh \
  output/test_cases/work1_qwen3_vl \
  --batch_size 32 \
  --max_tokens 512 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
```

<!-- For GLM-4.1V checkpoints:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 GLM41V_MODEL_PATH=/path/to/GLM-4.1V-9B-Thinking \
  ./run_generate_glm4_1v_checkpoints.sh \
  output/test_cases/work1_glm4_1v \
  --batch_size 16 \
  --max_tokens 1024 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
```

For LLaVA-1.5-7B, set the local model path and run:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 LLAVA15_MODEL_PATH=/path/to/llava-1.5-7b-hf \
  ./run_generate_llava_v1_5_7b_checkpoints.sh \
  output/test_cases/work1_llava_v1_5_7b \
  --batch_size 32 \
  --max_tokens 512 \
  --max_model_len 4096 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
```

For LLaVA-1.6-Vicuna-7B:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 LLAVA16_MODEL_PATH=/path/to/llava-v1.6-vicuna-7b-hf \
  ./run_generate_llava_v1_6_vicuna_checkpoints.sh \
  output/test_cases/work1_llava_v1_6_vicuna \
  --batch_size 32 \
  --max_tokens 512 \
  --max_model_len 4096 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
```

For Kimi-VL-A3B-Instruct:

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 KIMIVL_MODEL_PATH=/path/to/Kimi-VL-A3B-Instruct \
  ./run_generate_kimi_vl_a3b_checkpoints.sh \
  output/test_cases/work1_kimi_vl_a3b \
  --batch_size 16 \
  --max_tokens 512 \
  --max_model_len 8192 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --seed 42
``` -->

### 5. Evaluate ASR with HarmBench

```bash
conda activate work1-vllm
CUDA_VISIBLE_DEVICES=0,1 ./run_evaluate_llama_checkpoints.sh \
  output/test_cases/work1_qwen2_5_vl \
  --model /path/to/HarmBench-Llama-2-13b-cls \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90
```

The evaluator writes `evalaute_llama_iter_<iteration>.json` and prints:

```text
test_generate_cases_iter_<iteration>.jsonl: <successful>/<total>, ASR=<percentage>
```

For the additional models, evaluate their generated responses with the same
HarmBench classifier.

<!-- ```bash
conda activate work1-vllm
for checkpoint_dir in \
  output/test_cases/work1_llava_v1_5_7b \
  output/test_cases/work1_llava_v1_6_vicuna \
  output/test_cases/work1_kimi_vl_a3b; do
  CUDA_VISIBLE_DEVICES=0,1 ./run_evaluate_llama_checkpoints.sh \
    "${checkpoint_dir}" \
    --model /path/to/HarmBench-Llama-2-13b-cls \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90
done
``` -->

## Quantitative Results

### Cross-model transfer on HarmBench

Each cell reports ASR / SR (%). Bold and underlined values indicate the best and second-best results for each metric.

| Source model | Attack | Qwen2.5-VL | Qwen3-VL | GLM-4.1V | LLaVA-v1.5 | LLaVA-v1.6 | Kimi-VL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL | UMK | 58.0 / 40.0 | 11.5 / 7.1 | 7.0 / 4.2 | 30.0 / 20.3 | 26.5 / 17.5 | 12.0 / 7.4 |
| Qwen2.5-VL | BAP | 61.0 / 42.5 | 20.0 / 13.1 | 13.5 / 8.4 | 42.0 / 28.5 | 38.0 / 25.2 | 22.5 / 14.3 |
| Qwen2.5-VL | MUJA | 67.5 / 56.2 | 30.0 / 24.2 | 16.5 / 12.6 | 41.5 / 32.8 | 38.5 / 30.8 | 23.5 / 17.5 |
| Qwen2.5-VL | JPS | <ins>68.5</ins> / <ins>61.0</ins> | **42.0** / <ins>36.5</ins> | 22.5 / 18.4 | 51.0 / 43.3 | 46.5 / 39.2 | 31.0 / 25.8 |
| Qwen2.5-VL | UltraBreak | 65.0 / 56.0 | 34.5 / 28.2 | <ins>28.0</ins> / <ins>22.3</ins> | <ins>58.5</ins> / <ins>47.8</ins> | <ins>53.5</ins> / <ins>43.6</ins> | <ins>38.5</ins> / <ins>31.0</ins> |
| Qwen2.5-VL | FORCE | 53.5 / 39.5 | 12.5 / 8.2 | 7.5 / 4.8 | 24.0 / 16.7 | 20.5 / 13.8 | 10.0 / 6.2 |
| Qwen2.5-VL | Ours | **79.5** / **75.5** | <ins>41.5</ins> / **37.8** | **37.0** / **32.8** | **68.5** / **62.2** | **63.0** / **56.5** | **49.0** / **43.9** |
| Qwen3-VL | UMK | 19.5 / 12.1 | 40.5 / 27.0 | 9.5 / 5.8 | 28.5 / 18.9 | 25.0 / 16.1 | 13.5 / 8.2 |
| Qwen3-VL | BAP | 30.5 / 20.1 | 44.0 / 31.0 | 15.0 / 9.4 | 38.5 / 26.1 | 35.0 / 23.2 | 24.5 / 15.8 |
| Qwen3-VL | MUJA | 42.5 / 33.6 | 51.5 / 43.6 | 18.5 / 13.9 | 39.0 / 31.5 | 36.5 / 28.4 | 25.0 / 19.6 |
| Qwen3-VL | JPS | **54.5** / <ins>47.5</ins> | <ins>55.5</ins> / <ins>49.5</ins> | 24.0 / 19.7 | 47.5 / 40.1 | 43.0 / 36.1 | 33.5 / 28.1 |
| Qwen3-VL | UltraBreak | 47.5 / 39.2 | 52.0 / 43.5 | <ins>30.0</ins> / <ins>23.7</ins> | <ins>51.0</ins> / <ins>40.6</ins> | <ins>48.5</ins> / <ins>38.7</ins> | **45.5** / <ins>36.5</ins> |
| Qwen3-VL | FORCE | 18.0 / 11.7 | 34.5 / 23.5 | 9.5 / 5.9 | 21.5 / 14.3 | 19.0 / 12.2 | 11.5 / 6.9 |
| Qwen3-VL | Ours | <ins>53.5</ins> / **48.6** | **60.0** / **56.0** | **38.5** / **33.9** | **59.0** / **52.4** | **55.5** / **48.8** | <ins>44.5</ins> / **39.2** |
| GLM-4.1V | UMK | 10.0 / 6.1 | 5.0 / 3.0 | 43.5 / 30.0 | 25.5 / 17.1 | 22.0 / 14.3 | 11.5 / 7.1 |
| GLM-4.1V | BAP | 21.0 / 13.3 | 12.5 / 7.8 | 50.0 / 34.0 | 36.5 / 24.2 | 32.5 / 21.1 | 22.0 / 13.8 |
| GLM-4.1V | MUJA | 23.5 / 18.1 | 21.5 / 16.2 | 53.0 / 43.5 | 38.0 / 30.6 | 32.5 / 24.8 | 23.5 / 18.1 |
| GLM-4.1V | JPS | 31.0 / 25.6 | **31.5** / <ins>26.5</ins> | <ins>58.0</ins> / <ins>51.0</ins> | 45.5 / 38.0 | 41.0 / 34.0 | 30.0 / 24.8 |
| GLM-4.1V | UltraBreak | <ins>38.5</ins> / <ins>31.1</ins> | 27.0 / 21.3 | 56.5 / 47.0 | <ins>52.5</ins> / <ins>41.7</ins> | <ins>47.0</ins> / <ins>37.3</ins> | <ins>38.0</ins> / <ins>30.0</ins> |
| GLM-4.1V | FORCE | 8.5 / 5.4 | 4.5 / 2.7 | 40.0 / 28.0 | 18.5 / 12.3 | 16.0 / 10.3 | 8.0 / 4.8 |
| GLM-4.1V | Ours | **49.0** / **43.7** | <ins>30.5</ins> / **26.6** | **65.0** / **59.5** | **62.0** / **54.8** | **56.5** / **49.3** | **47.5** / **41.6** |
| LLaVA-v1.5 | UMK | 12.5 / 8.2 | 5.5 / 3.5 | 6.5 / 4.2 | 83.5 / 65.5 | 42.0 / 30.5 | 14.5 / 9.6 |
| LLaVA-v1.5 | BAP | 22.0 / 14.5 | 12.0 / 7.8 | 14.0 / 9.0 | 80.5 / 59.5 | 49.5 / 34.6 | 25.5 / 16.9 |
| LLaVA-v1.5 | MUJA | 24.5 / 18.8 | 14.0 / 10.6 | 17.0 / 13.4 | 84.5 / 72.2 | 58.0 / 47.8 | 26.0 / 20.4 |
| LLaVA-v1.5 | JPS | 31.5 / 26.2 | 19.5 / 16.0 | 22.0 / 18.0 | <ins>86.0</ins> / <ins>76.5</ins> | **68.0** / <ins>59.3</ins> | 34.5 / 28.9 |
| LLaVA-v1.5 | UltraBreak | **41.0** / <ins>33.2</ins> | <ins>26.0</ins> / <ins>20.4</ins> | <ins>28.5</ins> / <ins>22.3</ins> | 84.0 / 69.5 | 56.5 / 45.4 | <ins>41.5</ins> / <ins>32.9</ins> |
| LLaVA-v1.5 | FORCE | 4.5 / 3.0 | 2.0 / 1.2 | 3.0 / 1.9 | 82.0 / 65.0 | 27.5 / 19.5 | 5.5 / 3.5 |
| LLaVA-v1.5 | Ours | <ins>40.5</ins> / **35.7** | **30.0** / **26.2** | **33.0** / **28.5** | **94.0** / **86.5** | <ins>67.0</ins> / **59.5** | **48.0** / **42.2** |
| LLaVA-v1.6 | UMK | 14.0 / 9.0 | 6.0 / 3.7 | 7.0 / 4.4 | 45.5 / 33.0 | 75.0 / 56.0 | 16.5 / 10.8 |
| LLaVA-v1.6 | BAP | 25.0 / 16.4 | 13.5 / 8.7 | 16.0 / 10.2 | 52.0 / 36.3 | 77.5 / 55.5 | 28.5 / 18.7 |
| LLaVA-v1.6 | MUJA | 26.0 / 20.2 | 16.0 / 12.5 | 16.5 / 12.8 | 58.5 / 47.4 | 78.0 / 65.7 | 28.0 / 21.2 |
| LLaVA-v1.6 | JPS | 34.0 / 28.3 | 21.5 / 17.6 | 24.5 / 20.0 | **70.5** / <ins>61.2</ins> | 79.5 / <ins>70.0</ins> | 37.0 / 31.0 |
| LLaVA-v1.6 | UltraBreak | <ins>41.0</ins> / <ins>33.3</ins> | <ins>28.0</ins> / <ins>22.1</ins> | <ins>31.5</ins> / <ins>24.9</ins> | 60.5 / 48.7 | <ins>81.0</ins> / 67.5 | <ins>44.0</ins> / <ins>35.2</ins> |
| LLaVA-v1.6 | FORCE | 5.5 / 3.5 | 2.5 / 1.5 | 4.0 / 2.5 | 30.5 / 21.7 | 76.5 / 57.5 | 6.5 / 4.0 |
| LLaVA-v1.6 | Ours | **47.0** / **41.6** | **32.0** / **27.8** | **36.0** / **31.1** | <ins>69.5</ins> / **61.7** | **90.5** / **82.0** | **51.0** / **44.7** |
| Kimi-VL | UMK | 13.0 / 8.0 | 6.5 / 3.9 | 8.0 / 4.9 | 27.0 / 18.0 | 24.0 / 15.6 | 50.0 / 34.0 |
| Kimi-VL | BAP | 24.0 / 15.4 | 15.0 / 9.6 | 16.5 / 10.3 | 39.0 / 26.1 | 36.0 / 23.6 | 60.5 / 41.5 |
| Kimi-VL | MUJA | 25.0 / 19.3 | 17.5 / 13.2 | 24.0 / 18.8 | 37.5 / 29.8 | 35.0 / 27.1 | 61.0 / 50.6 |
| Kimi-VL | JPS | 34.0 / 28.4 | 23.5 / 19.4 | **36.5** / **30.9** | 47.5 / 40.0 | 44.5 / 37.3 | 66.0 / <ins>58.5</ins> |
| Kimi-VL | UltraBreak | <ins>41.0</ins> / <ins>33.2</ins> | <ins>30.0</ins> / <ins>23.8</ins> | 32.5 / 25.7 | <ins>55.0</ins> / <ins>43.8</ins> | <ins>51.5</ins> / <ins>41.1</ins> | <ins>67.5</ins> / 56.0 |
| Kimi-VL | FORCE | 11.0 / 6.9 | 6.0 / 3.6 | 8.5 / 5.2 | 20.0 / 13.1 | 18.0 / 11.4 | 49.5 / 33.0 |
| Kimi-VL | Ours | **51.0** / **45.6** | **39.0** / **34.6** | <ins>35.5</ins> / <ins>30.8</ins> | **66.0** / **58.6** | **61.0** / **53.6** | **76.5** / **70.0** |

## Qualitative Results

### Universal adversarial examples

![](figs/qualitative-success.png)


## Citation

```bibtex
@inproceedings{anonymous2026work1,
  author    = {},
  title     = {},
  booktitle = {},
  year      = {2026}
}
```


## Contact

If you have any questions, please feel free to contact Mr. Lijie Shang via email (sljcs@hdu.edu.cn).


## Acknowledgement

We thank the authors of the open-source target models and HarmBench-Llama-2-13b-cls.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
