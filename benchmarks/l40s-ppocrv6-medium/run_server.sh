#!/usr/bin/env bash
# Launch turboocr-server as used for every measurement in this directory.
# Run from anywhere; paths resolve relative to the repo root.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# TensorRT is dlopen'd at runtime and is not on the default loader path.
export LD_LIBRARY_PATH="${TENSORRT_DIR:-/usr/local/tensorrt}/lib:/usr/local/cuda/lib64:${REPO}/third_party/onnxruntime/lib:${LD_LIBRARY_PATH:-}"

export OCR_MODEL="${OCR_MODEL:-medium}"
export LOG_FORMAT="${LOG_FORMAT:-text}"

# First start builds TensorRT engines from ONNX into ~/.cache/turbo-ocr and can
# take hours on non-Blackwell GPUs (see README). Set TRT_OPT_LEVEL=3 to trade a
# small amount of steady-state speed for a much shorter build.
cd "$REPO"
exec ./build/turboocr-server "$@"
