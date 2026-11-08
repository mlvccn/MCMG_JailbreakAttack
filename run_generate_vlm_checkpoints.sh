#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s MODEL_NAME MODEL_PATH CHECKPOINT_DIR [vLLM generation options...]\n' "$0"
    exit 0
fi
if [[ $# -lt 3 ]]; then
    printf 'Usage: %s MODEL_NAME MODEL_PATH CHECKPOINT_DIR [vLLM generation options...]\n' "$0" >&2
    exit 2
fi
model_name="$1"
model_path="$2"
checkpoint_dir="$3"
shift 3

if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint_dir}" >&2
    exit 1
fi
checkpoint_dir="$(cd -- "${checkpoint_dir}" && pwd)"
cd -- "${script_dir}"

shopt -s nullglob
input_files=("${checkpoint_dir}"/test_cases_iter_*.jsonl)
if [[ ${#input_files[@]} -eq 0 ]]; then
    echo "No test_cases_iter_*.jsonl files found in ${checkpoint_dir}" >&2
    echo "Run generate_work1_checkpoint_cases.py first." >&2
    exit 1
fi
mapfile -d '' -t input_files < <(printf '%s\0' "${input_files[@]}" | sort -z -V)

command=(python "${script_dir}/generate_qwen2_5_slj.py"
    --model_path "${model_path}"
    --model_name "${model_name}")
pending=0
for input_file in "${input_files[@]}"; do
    filename="${input_file##*/}"
    iteration="${filename#test_cases_iter_}"
    iteration="${iteration%.jsonl}"
    output_file="${checkpoint_dir}/test_generate_cases_iter_${iteration}.jsonl"
    input_count="$(wc -l < "${input_file}")"
    output_count=0
    if [[ -f "${output_file}" ]]; then
        output_count="$(wc -l < "${output_file}")"
    fi
    if [[ "${input_count}" -eq "${output_count}" && "${input_count}" -gt 0 ]]; then
        echo "Skipping iteration ${iteration}: output already exists"
        continue
    fi
    command+=(--input_jsonl "${input_file}" --output_jsonl "${output_file}")
    pending=$((pending + 1))
done
if [[ "${pending}" -eq 0 ]]; then
    echo "All iteration outputs are complete"
    exit 0
fi
echo "Inference Python: $(command -v python)"
"${command[@]}" "$@"
