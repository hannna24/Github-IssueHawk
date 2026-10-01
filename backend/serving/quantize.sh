#!/usr/bin/env bash
# Convert the merged fp16 model to GGUF and quantize it for CPU serving.
# Run on Linux (e.g. Colab) -- the merged model lives there, not on the laptop.
#
#   bash quantize.sh [MERGED_DIR] [OUT_DIR] [QUANT]
#   bash quantize.sh ../checkpoints/merged ../checkpoints/gguf Q4_K_M
#
# Produces OUT_DIR/issuehawk-f16.gguf (~6 GB, intermediate) and
# OUT_DIR/issuehawk-<QUANT>.gguf (~2 GB for Q4_K_M).
set -euo pipefail

MERGED_DIR="${1:-../checkpoints/merged}"
OUT_DIR="${2:-../checkpoints/gguf}"
QUANT="${3:-Q4_K_M}"
LLAMA_DIR="${LLAMA_DIR:-./llama.cpp}"

[ -f "$MERGED_DIR/config.json" ] || { echo "no merged model at $MERGED_DIR"; exit 1; }
mkdir -p "$OUT_DIR"

if [ ! -d "$LLAMA_DIR" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# llama.cpp dropped the plain `make` build; CMake is the supported path.
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)"

pip install -q gguf sentencepiece protobuf

F16="$OUT_DIR/issuehawk-f16.gguf"
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$F16" --outtype f16
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$OUT_DIR/issuehawk-$QUANT.gguf" "$QUANT"

ls -lh "$OUT_DIR"
echo "done: $OUT_DIR/issuehawk-$QUANT.gguf"
