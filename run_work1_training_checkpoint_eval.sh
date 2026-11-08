#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [checkpoint_dir]\n' "$0"
    printf 'Environment overrides: PYTHON_BIN, VLLM_PYTHON, QWEN_MODEL_PATH, '
    printf 'HARMBENCH_MODEL_PATH, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION.\n'
    exit 0
fi

checkpoint_dir="${1:-${repo_dir}/output/test_cases/work1_qwen2_5_vl}"
training_dir="${checkpoint_dir}/training_eval"

python_bin="${PYTHON_BIN:-python}"
vllm_python="${VLLM_PYTHON:-python}"
qwen_model="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
harmbench_model="${HARMBENCH_MODEL_PATH:-cais/HarmBench-Llama-2-13b-cls}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-2}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
generation_batch_size="${GENERATION_BATCH_SIZE:-32}"
generation_max_tokens="${GENERATION_MAX_TOKENS:-512}"
generation_max_model_len="${GENERATION_MAX_MODEL_LEN:-8192}"
classifier_max_model_len="${CLASSIFIER_MAX_MODEL_LEN:-2048}"

if ! command -v "${python_bin}" >/dev/null 2>&1 && [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi
if ! command -v "${vllm_python}" >/dev/null 2>&1 && [[ ! -x "${vllm_python}" ]]; then
    echo "vLLM Python executable not found: ${vllm_python}" >&2
    exit 1
fi
if ! "${vllm_python}" -c "import vllm"; then
    echo "vLLM is unavailable in: ${vllm_python}" >&2
    exit 1
fi
if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint_dir}" >&2
    exit 1
fi

cd "${repo_dir}"

"${python_bin}" generate_work1_training_checkpoint_cases.py \
    "${checkpoint_dir}" \
    --output-dir "${training_dir}" \
    --iterations 500 1000 1500 2000

generation_command=(
    "${vllm_python}" generate_qwen2_5_slj.py
    --model_path "${qwen_model}"
    --batch_size "${generation_batch_size}"
    --max_tokens "${generation_max_tokens}"
    --tensor_parallel_size "${tensor_parallel_size}"
    --gpu_memory_utilization "${gpu_memory_utilization}"
    --max_model_len "${generation_max_model_len}"
)
generation_pending=0
for iteration in 500 1000 1500 2000; do
    input_file="${training_dir}/test_cases_iter_${iteration}.jsonl"
    output_file="${training_dir}/test_generate_cases_iter_${iteration}.jsonl"
    input_count="$(wc -l < "${input_file}")"
    output_count=0
    if [[ -f "${output_file}" ]]; then
        output_count="$(wc -l < "${output_file}")"
    fi
    if [[ "${input_count}" -eq "${output_count}" && "${input_count}" -gt 0 ]]; then
        echo "Skipping generation iteration ${iteration}: ${output_count} responses exist"
        continue
    fi
    generation_command+=(--input_jsonl "${input_file}" --output_jsonl "${output_file}")
    generation_pending=$((generation_pending + 1))
done

if [[ "${generation_pending}" -gt 0 ]]; then
    "${generation_command[@]}"
fi

evaluation_command=(
    "${vllm_python}" evaluate_qwen2_5_llama2_slj.py
)
evaluation_inputs=()
evaluation_outputs=()
evaluation_pending=0
for iteration in 500 1000 1500 2000; do
    input_file="${training_dir}/test_generate_cases_iter_${iteration}.jsonl"
    output_file="${training_dir}/evalaute_llama_iter_${iteration}.json"
    if [[ -s "${output_file}" ]]; then
        echo "Skipping evaluation iteration ${iteration}: result exists"
        continue
    fi
    evaluation_inputs+=("${input_file}")
    evaluation_outputs+=("${output_file}")
    evaluation_pending=$((evaluation_pending + 1))
done

if [[ "${evaluation_pending}" -gt 0 ]]; then
    evaluation_command+=("${evaluation_inputs[@]}")
    for output_file in "${evaluation_outputs[@]}"; do
        evaluation_command+=(--output-file "${output_file}")
    done
    evaluation_command+=(
        --model "${harmbench_model}"
        --tensor-parallel-size "${tensor_parallel_size}"
        --gpu-memory-utilization "${gpu_memory_utilization}"
        --max-model-len "${classifier_max_model_len}"
    )
    "${evaluation_command[@]}"
fi

"${python_bin}" summarize_work1_training_eval.py \
    "${checkpoint_dir}" \
    --training-dir "${training_dir}"
