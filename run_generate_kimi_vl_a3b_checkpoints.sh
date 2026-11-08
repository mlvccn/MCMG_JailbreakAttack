#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [checkpoint_dir] [vLLM generation options...]\n' "$0"
    printf 'Model path override: KIMIVL_MODEL_PATH\n'
    exit 0
fi
checkpoint_dir="${script_dir}/output/test_cases/work1_kimi_vl_a3b"
if [[ $# -gt 0 && "$1" != -* ]]; then checkpoint_dir="$1"; shift; fi
exec "${script_dir}/run_generate_vlm_checkpoints.sh" \
  "kimi_vl_a3b" "${KIMIVL_MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}" "${checkpoint_dir}" "$@"
