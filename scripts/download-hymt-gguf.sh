#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download-hymt-gguf.sh [quant_or_filename] [output_dir]

Defaults:
  quant_or_filename  Q4_K_M
  output_dir         models/translate

Examples:
  scripts/download-hymt-gguf.sh
  scripts/download-hymt-gguf.sh Q6_K
  scripts/download-hymt-gguf.sh HY-MT1.5-1.8B-Q8_0.gguf models/translate

Downloads Tencent HY-MT1.5-1.8B GGUF from Hugging Face. The Q4_K_M file is
about 1.13 GB.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_id="tencent/HY-MT1.5-1.8B-GGUF"
quant_or_filename="${1:-Q4_K_M}"
output_dir="${2:-models/translate}"

if [[ "$quant_or_filename" == *.gguf ]]; then
  filename="$quant_or_filename"
else
  filename="HY-MT1.5-1.8B-${quant_or_filename}.gguf"
fi

url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
target="${output_dir}/${filename}"

mkdir -p "$output_dir"

if [[ -f "$target" ]]; then
  echo "Model already exists: $target"
  exit 0
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

echo "Done: $target"
