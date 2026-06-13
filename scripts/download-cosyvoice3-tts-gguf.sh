#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download-cosyvoice3-tts-gguf.sh [output_dir] [llm_quant] [flow_quant]

Defaults:
  output_dir  models/tts/cosyvoice3
  llm_quant   q4_k
  flow_quant  q8_0

Examples:
  scripts/download-cosyvoice3-tts-gguf.sh
  scripts/download-cosyvoice3-tts-gguf.sh models/tts/cosyvoice3 f16 f16

Downloads the CrispASR CosyVoice3 GGUF files needed for baked-voice TTS from
cstr/cosyvoice3-0.5b-2512-GGUF. The default minimum viable combo is about
745 MB:

  cosyvoice3-llm-q4_k.gguf
  cosyvoice3-flow-q8_0.gguf
  cosyvoice3-hift-f16.gguf
  cosyvoice3-voices.gguf

For arbitrary WAV voice cloning, also download the s3tok and campplus companion
models from the same Hugging Face repo.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

output_dir="${1:-models/tts/cosyvoice3}"
llm_quant="${2:-q4_k}"
flow_quant="${3:-q8_0}"
repo_id="cstr/cosyvoice3-0.5b-2512-GGUF"

mkdir -p "$output_dir"

download_one() {
  local filename="$1"
  local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
  local target="${output_dir}/${filename}"
  local partial="${target}.part"

  if [[ -f "$target" ]]; then
    echo "File already exists: $target"
    return
  fi

  echo "Downloading ${repo_id}/${filename}"
  echo "Target: $target"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$partial" "$url"
  else
    echo "Missing curl or wget." >&2
    exit 1
  fi

  mv -f "$partial" "$target"
}

download_one "cosyvoice3-llm-${llm_quant}.gguf"
download_one "cosyvoice3-flow-${flow_quant}.gguf"
download_one "cosyvoice3-hift-f16.gguf"
download_one "cosyvoice3-voices.gguf"

echo "Done:"
echo "  ${output_dir}/cosyvoice3-llm-${llm_quant}.gguf"
echo "  ${output_dir}/cosyvoice3-flow-${flow_quant}.gguf"
echo "  ${output_dir}/cosyvoice3-hift-f16.gguf"
echo "  ${output_dir}/cosyvoice3-voices.gguf"
