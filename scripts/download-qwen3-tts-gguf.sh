#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download-qwen3-tts-gguf.sh [output_dir] [variant] [quant]

Defaults:
  output_dir  models/tts/qwen3-tts-0.6b-customvoice
  variant     customvoice
  quant       q8_0

Variants:
  customvoice  Qwen3-TTS-12Hz-0.6B-CustomVoice, preset speakers, no ref WAV
  base         Qwen3-TTS-12Hz-0.6B-Base, requires --tts-voice-model voice GGUF or WAV + --tts-ref-text

Examples:
  scripts/download-qwen3-tts-gguf.sh
  scripts/download-qwen3-tts-gguf.sh models/tts/qwen3-tts-0.6b-base base q8_0

Downloads the Qwen3-TTS 0.6B talker and the required tokenizer/codec GGUF:

  qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf
  qwen3-tts-tokenizer-12hz.gguf
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

output_dir="${1:-models/tts/qwen3-tts-0.6b-customvoice}"
variant="${2:-customvoice}"
quant="${3:-q8_0}"

case "$variant" in
  customvoice|custom-voice|cv)
    repo_id="cstr/qwen3-tts-0.6b-customvoice-GGUF"
    model_filename="qwen3-tts-12hz-0.6b-customvoice-${quant}.gguf"
    ;;
  base)
    repo_id="cstr/qwen3-tts-0.6b-base-GGUF"
    model_filename="qwen3-tts-12hz-0.6b-base-${quant}.gguf"
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    usage >&2
    exit 1
    ;;
esac

codec_repo_id="cstr/qwen3-tts-tokenizer-12hz-GGUF"
codec_filename="qwen3-tts-tokenizer-12hz.gguf"

mkdir -p "$output_dir"

download_one() {
  local repo="$1"
  local filename="$2"
  local url="https://huggingface.co/${repo}/resolve/main/${filename}"
  local target="${output_dir}/${filename}"
  local partial="${target}.part"

  if [[ -f "$target" ]]; then
    echo "File already exists: $target"
    return
  fi

  echo "Downloading ${repo}/${filename}"
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

download_one "$repo_id" "$model_filename"
download_one "$codec_repo_id" "$codec_filename"

echo "Done:"
echo "  ${output_dir}/${model_filename}"
echo "  ${output_dir}/${codec_filename}"
