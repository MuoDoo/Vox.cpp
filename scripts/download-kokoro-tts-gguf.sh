#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download-kokoro-tts-gguf.sh [output_dir] [model_quant] [voice_name]

Defaults:
  output_dir   models/tts/kokoro
  model_quant  q8_0
  voice_name   af_heart

Examples:
  scripts/download-kokoro-tts-gguf.sh
  scripts/download-kokoro-tts-gguf.sh models/tts/kokoro q8_0 af_bella

Downloads the CrispASR Kokoro GGUF files needed for TTS:

  kokoro-82m-q8_0.gguf
  kokoro-voice-af_heart.gguf

Kokoro uses espeak-ng for phonemization. Install espeak-ng or ensure the
espeak-ng executable is on PATH. When libespeak-ng is available, the runtime
loads it dynamically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

output_dir="${1:-models/tts/kokoro}"
model_quant="${2:-q8_0}"
voice_name="${3:-af_heart}"

mkdir -p "$output_dir"

download_one() {
  local repo_id="$1"
  local filename="$2"
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

download_one "cstr/kokoro-82m-GGUF" "kokoro-82m-${model_quant}.gguf"
download_one "cstr/kokoro-voices-GGUF" "kokoro-voice-${voice_name}.gguf"

echo "Done:"
echo "  ${output_dir}/kokoro-82m-${model_quant}.gguf"
echo "  ${output_dir}/kokoro-voice-${voice_name}.gguf"
