#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="${script_dir}/output/test_cases/work1_glm4_1v"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [checkpoint_dir] [vLLM generation options...]\n' "$0"
    printf 'Default directory: output/test_cases/work1_glm4_1v\n'
    printf 'Set GLM41V_MODEL_PATH to override the default model.\n'
    exit 0
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint_dir="$1"
    shift
fi
if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint_dir}" >&2
    exit 1
fi
checkpoint_dir="$(cd -- "${checkpoint_dir}" && pwd)"
cd -- "${script_dir}"

shopt -s nullglob
conversion_needed=0
for checkpoint in "${checkpoint_dir}"/checkpoint_iter_*.pt; do
    filename="${checkpoint##*/}"
    iteration="${filename#checkpoint_iter_}"
    iteration="${iteration%.pt}"
    if [[ ! -f "${checkpoint_dir}/test_cases_iter_${iteration}.jsonl" ||
          ! -f "${checkpoint_dir}/work1_universal_iter_${iteration}.png" ]]; then
        conversion_needed=1
        break
    fi
done
if [[ "${conversion_needed}" -eq 1 ]]; then
    project_python="${WORK1_PYTHON:-python}"
    echo "Converting checkpoints into per-iteration test cases"
    "${project_python}" "${script_dir}/generate_work1_checkpoint_cases.py" "${checkpoint_dir}"
fi

input_files=("${checkpoint_dir}"/test_cases_iter_*.jsonl)
if [[ ${#input_files[@]} -eq 0 ]]; then
    echo "No test_cases_iter_*.jsonl or convertible checkpoints found in ${checkpoint_dir}" >&2
    exit 1
fi
mapfile -d '' -t input_files < <(printf '%s\0' "${input_files[@]}" | sort -z -V)

command=(python "${script_dir}/generate_qwen2_5_slj.py"
    --model_path "${GLM41V_MODEL_PATH:-zai-org/GLM-4.1V-9B-Thinking}"
    --model_name glm4_1v
    --max_model_len 8192)
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
        echo "Skipping iteration ${iteration}: ${output_file} already exists"
        continue
    fi
    echo "Queued iteration ${iteration}"
    command+=(--input_jsonl "${input_file}" --output_jsonl "${output_file}")
    pending=$((pending + 1))
done
if [[ "${pending}" -eq 0 ]]; then
    echo "All iteration outputs are complete"
    exit 0
fi

echo "Inference Python: $(command -v python)"
"${command[@]}" "$@"
