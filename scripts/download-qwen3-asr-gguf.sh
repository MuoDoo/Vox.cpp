#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download-qwen3-asr-gguf.sh [size] [quant] [output_dir]

Defaults:
  size        1.7B
  quant       Q8_0
  output_dir  models/asr/qwen3-asr-1.7b

Examples:
  scripts/download-qwen3-asr-gguf.sh
  scripts/download-qwen3-asr-gguf.sh 0.6B Q8_0 models/asr/qwen3-asr-0.6b
  scripts/download-qwen3-asr-gguf.sh 1.7B bf16 models/asr/qwen3-asr-1.7b-bf16

Downloads the llama.cpp GGUF Qwen3-ASR model and matching mmproj from
ggml-org/Qwen3-ASR-${size}-GGUF. For 1.7B Q8_0, the model is about 2.17 GB
and the mmproj is about 356 MB.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

size="${1:-1.7B}"
quant="${2:-Q8_0}"
size_lower="$(printf '%s' "$size" | tr '[:upper:]' '[:lower:]')"
default_dir="models/asr/qwen3-asr-${size_lower}"
output_dir="${3:-$default_dir}"

repo_id="ggml-org/Qwen3-ASR-${size}-GGUF"
model_filename="Qwen3-ASR-${size}-${quant}.gguf"
mmproj_filename="mmproj-Qwen3-ASR-${size}-${quant}.gguf"

mkdir -p "$output_dir"

download_one() {
  local filename="$1"
  local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
  local target="${output_dir}/${filename}"

  if [[ -f "$target" ]]; then
    echo "File already exists: $target"
    return
  fi

  echo "Downloading ${repo_id}/${filename}"
  echo "Target: $target"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "$target" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$target" "$url"
  else
    echo "Missing curl or wget." >&2
    exit 1
  fi
}

download_one "$model_filename"
download_one "$mmproj_filename"

echo "Done:"
echo "  ${output_dir}/${model_filename}"
echo "  ${output_dir}/${mmproj_filename}"
