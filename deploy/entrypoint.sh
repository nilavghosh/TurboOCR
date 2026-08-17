#!/usr/bin/env bash
# TurboOCR container entrypoint: resolve the right TensorRT engine cache out of
# a shared /models volume, then start the server (optionally behind the
# PaddleX-compatible adapter).
#
# WHY A VARIANT DIRECTORY
# A cached engine is only reusable on a host whose GPU architecture, CUDA
# driver, CUDA runtime and TensorRT version all match the machine that built it
# (src/engine/trt/trt_engine_cache.cpp folds all four into the cache key). Put
# engines from two different machines in one flat directory and nothing breaks
# -- the filename hash differs, so they simply coexist -- but you lose any way
# to tell which is which, and a cache miss silently rebuilds for hours instead
# of failing. So engines live under a directory named for the combination that
# produced them:
#
#   /models/engines/sm89-rt12.4-drv12.4-trt10.16/
#
# The model TIER is deliberately NOT in that path. det/rec engines for tiny,
# small and medium already hash differently, so all three coexist safely in one
# variant directory and OCR_MODEL alone selects between them at runtime.
#
# WHAT CAN AND CANNOT BE SET AT RUNTIME
#   OCR_MODEL          runtime   tiny | small | medium (+ other scripts)
#   GPU_DEVICE         runtime   which physical GPU to bind
#   GPU_ARCH           runtime   override the detected sm_XX (rarely needed)
#   TRT_OPT_LEVEL      runtime   part of the cache key -- changing it re-builds
#   CUDA runtime ver.  BUILD     compiled into the binary; a build arg, not env
#   TensorRT version   BUILD     baked into the image; a build arg, not env
#   GPU architecture   BUILD*    CMAKE_CUDA_ARCHITECTURES is compile-time, but
#                                embedded PTX lets a binary JIT onto newer
#                                cards; the engine itself is always per-arch
#
# The CUDA and TensorRT values below are stamped in at build time and are only
# reported here -- setting them as env vars would rename the directory without
# changing what the binary actually links, which is how you end up mounting an
# incompatible cache and rebuilding anyway.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*"; }
die() { printf '[entrypoint] ERROR: %s\n' "$*" >&2; exit 1; }

MODELS_DIR="${MODELS_DIR:-/models}"
OCR_MODEL="${OCR_MODEL:-medium}"
export OCR_MODEL

# ---- GPU selection -------------------------------------------------------
# GPU_DEVICE picks the physical card on a multi-GPU host. Set before any CUDA
# call so architecture detection reads the card we will actually run on.
if [[ -n "${GPU_DEVICE:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    log "GPU_DEVICE=$GPU_DEVICE -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- run with --gpus"

# ---- resolve the variant -------------------------------------------------
# Compute capability of the selected GPU, e.g. "8.9" -> sm89.
if [[ -n "${GPU_ARCH:-}" ]]; then
    ARCH="${GPU_ARCH#sm}"; ARCH="${ARCH//./}"
    log "GPU_ARCH override: sm${ARCH}"
else
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    [[ -n "$CC" ]] || die "could not read compute_cap from nvidia-smi; set GPU_ARCH (e.g. GPU_ARCH=89)"
    ARCH="${CC//./}"
fi

# The driver's *reported* CUDA version is what lands in the cache key, not the
# driver build string: 550.144.03 and 550.90 both report 12.4 and share engines.
DRV_CUDA="${DRIVER_CUDA:-$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)}"
[[ -n "$DRV_CUDA" ]] || DRV_CUDA="unknown"

# Stamped at build time (see Dockerfile ARGs). Never derive these from env.
RT_CUDA="${TURBO_CUDA_RUNTIME:-unknown}"
TRT_VER="${TURBO_TRT_VERSION:-unknown}"
TRT_MM="$(echo "$TRT_VER" | cut -d. -f1,2)"

VARIANT="${TURBO_ENGINE_VARIANT:-sm${ARCH}-rt${RT_CUDA}-drv${DRV_CUDA}-trt${TRT_MM}}"
ENGINE_DIR="${MODELS_DIR}/engines/${VARIANT}"

export TRT_ENGINE_CACHE="$ENGINE_DIR"

log "GPU            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (sm${ARCH})"
log "CUDA           driver ${DRV_CUDA} · runtime ${RT_CUDA} (build-time)"
log "TensorRT       ${TRT_VER} (build-time)"
log "OCR_MODEL      ${OCR_MODEL}"
log "engine cache   ${ENGINE_DIR}"

if [[ ! -d "$MODELS_DIR" ]]; then
    die "$MODELS_DIR does not exist -- mount the models volume, e.g. -v turboocr-models:$MODELS_DIR"
fi

# A read-only mount is a legitimate production choice: it turns a cache miss
# into a loud failure instead of a silent multi-hour rebuild.
CACHE_WRITABLE=1
if ! mkdir -p "$ENGINE_DIR" 2>/dev/null; then
    CACHE_WRITABLE=0
    [[ -d "$ENGINE_DIR" ]] || die "$ENGINE_DIR is missing and $MODELS_DIR is not writable.
  Either mount the volume read-write for the first run, or pre-populate:
    <volume>/engines/${VARIANT}/*.trt"
fi
[[ -w "$ENGINE_DIR" ]] || CACHE_WRITABLE=0

ENGINE_COUNT="$(find "$ENGINE_DIR" -maxdepth 1 -name '*.trt' 2>/dev/null | wc -l | tr -d ' ')"
log "engines found  ${ENGINE_COUNT}"

if [[ "$ENGINE_COUNT" -eq 0 ]]; then
    if [[ "$CACHE_WRITABLE" -eq 0 ]]; then
        die "no engines in $ENGINE_DIR and the cache is read-only.
  This host needs a cache built for variant: ${VARIANT}
  Build one with:  docker run --rm -e WARM_ONLY=1 -v <vol>:${MODELS_DIR} <image>"
    fi
    log "WARNING: cache is empty -- TensorRT will build engines from ONNX."
    log "         Expect hours on non-Blackwell GPUs at TRT_OPT_LEVEL=${TRT_OPT_LEVEL:-5}."
    log "         Set TRT_OPT_LEVEL=3 to cut that substantially."
fi

# Record what produced this directory, and warn when the mounted cache was
# built somewhere incompatible -- the failure is otherwise silent.
MANIFEST="${ENGINE_DIR}/VARIANT"
STAMP="arch=sm${ARCH} driver_cuda=${DRV_CUDA} runtime_cuda=${RT_CUDA} tensorrt=${TRT_VER}"
if [[ -f "$MANIFEST" ]]; then
    PREV="$(cat "$MANIFEST" 2>/dev/null || true)"
    if [[ "$PREV" != "$STAMP" ]]; then
        log "WARNING: cache manifest mismatch -- engines here will be ignored and rebuilt"
        log "         found:    ${PREV}"
        log "         expected: ${STAMP}"
    fi
elif [[ "$CACHE_WRITABLE" -eq 1 ]]; then
    printf '%s\n' "$STAMP" > "$MANIFEST" 2>/dev/null || true
fi

# ---- optional: build engines and exit ------------------------------------
# Pre-bakes a cache into the volume without serving traffic, so the multi-hour
# build happens once in a job rather than on a pod that is meant to be ready.
if [[ "${WARM_ONLY:-0}" == "1" ]]; then
    log "WARM_ONLY=1 -- building engines for OCR_MODEL=${OCR_MODEL}, then exiting"
    ./build/turboocr-server --http-port "${TURBO_PORT:-8081}" &
    SRV=$!
    while ! curl -fsS "http://127.0.0.1:${TURBO_PORT:-8081}/health/ready" >/dev/null 2>&1; do
        kill -0 "$SRV" 2>/dev/null || die "server exited during engine build"
        sleep 5
    done
    log "engines ready in ${ENGINE_DIR}:"
    find "$ENGINE_DIR" -maxdepth 1 -name '*.trt' -printf '  %f (%s bytes)\n' 2>/dev/null || true
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    exit 0
fi

# ---- serve ---------------------------------------------------------------
# Compat adapters (PaddleX, Triton/KServe v2) run in front of the C++ server:
# the adapter owns the public port and turboocr-server binds TURBO_PORT
# privately. Exactly one may be enabled — they would otherwise fight over the
# same listener, and the second one to bind would fail long after the first
# looked healthy.
ADAPTER_MODULE=""
if [[ "${PADDLEX_API:-0}" == "1" && "${TRITON_API:-0}" == "1" ]]; then
    die "PADDLEX_API=1 and TRITON_API=1 are mutually exclusive -- run two containers, one per protocol"
fi
if [[ "${PADDLEX_API:-0}" == "1" ]]; then
    ADAPTER_NAME="PaddleX"
    ADAPTER_MODULE="paddlex_adapter:app"
    ADAPTER_DIR=/app/compat/paddlex
    ADAPTER_PORT="${PADDLEX_PORT:-8080}"
    ADAPTER_WORKERS="${PADDLEX_WORKERS:-4}"
    ADAPTER_LOG="${PADDLEX_LOG_LEVEL:-info}"
elif [[ "${TRITON_API:-0}" == "1" ]]; then
    ADAPTER_NAME="Triton/KServe v2"
    ADAPTER_MODULE="triton_adapter:app"
    ADAPTER_DIR=/app/compat/triton
    # 8000 is Triton's own default HTTP port, so a client moves with a
    # hostname change and nothing else.
    ADAPTER_PORT="${TRITON_PORT:-8000}"
    ADAPTER_WORKERS="${TRITON_WORKERS:-4}"
    ADAPTER_LOG="${TRITON_LOG_LEVEL:-info}"
fi

if [[ -n "$ADAPTER_MODULE" ]]; then
    TURBO_PORT="${TURBO_PORT:-8081}"
    export TURBO_OCR_URL="http://127.0.0.1:${TURBO_PORT}"

    # Kill the whole process group on exit: a crash in either process should
    # take the container down, not leave a half-serving zombie that still
    # passes a TCP check.
    _shutdown() { trap - TERM INT EXIT; kill -TERM -$$ 2>/dev/null || true; }
    trap _shutdown TERM INT EXIT

    ./build/turboocr-server --http-port "${TURBO_PORT}" &
    SRV=$!
    log "waiting for backend on :${TURBO_PORT}"
    while ! curl -fsS "http://127.0.0.1:${TURBO_PORT}/health/ready" >/dev/null 2>&1; do
        kill -0 "$SRV" 2>/dev/null || die "turboocr-server exited during startup"
        sleep 2
    done
    log "backend ready; ${ADAPTER_NAME}-compatible API on :${ADAPTER_PORT}"
    exec uvicorn "${ADAPTER_MODULE}" \
        --app-dir "${ADAPTER_DIR}" \
        --host 0.0.0.0 --port "${ADAPTER_PORT}" \
        --workers "${ADAPTER_WORKERS}" \
        --log-level "${ADAPTER_LOG}"
fi

exec ./build/turboocr-server --http-port "${TURBO_PORT:-8080}" "$@"
