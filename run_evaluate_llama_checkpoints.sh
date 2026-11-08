#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="${script_dir}/output/test_cases/work1_qwen2_5_vl"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [checkpoint_dir] [HarmBench evaluation options...]\n' "$0"
    printf 'Default directory: output/test_cases/work1_qwen2_5_vl\n'
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
generated_files=("${checkpoint_dir}"/test_generate_cases_iter_*.jsonl)
if [[ ${#generated_files[@]} -eq 0 ]]; then
    echo "No test_generate_cases_iter_*.jsonl files found in ${checkpoint_dir}" >&2
    exit 1
fi

mapfile -d '' -t generated_files < <(printf '%s\0' "${generated_files[@]}" | sort -z -V)

pending_files=()
for input_file in "${generated_files[@]}"; do
    filename="${input_file##*/}"
    iteration="${filename#test_generate_cases_iter_}"
    iteration="${iteration%.jsonl}"
    output_file="${checkpoint_dir}/evalaute_llama_iter_${iteration}.json"

    if [[ -s "${output_file}" ]]; then
        echo "Skipping iteration ${iteration}: ${output_file} already exists"
        continue
    fi
    echo "Queued iteration ${iteration}"
    pending_files+=("${input_file}")
done

if [[ ${#pending_files[@]} -eq 0 ]]; then
    echo "All iteration evaluations are complete"
    exit 0
fi

python "${script_dir}/evaluate_qwen2_5_llama2_slj.py" "${pending_files[@]}" "$@"
