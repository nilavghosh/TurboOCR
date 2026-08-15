# Deploying with a shared `/models` volume

One volume, many machines. Mount `/models`, and the container works out which
TensorRT engines belong to the GPU it landed on.

```bash
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models \
  -e OCR_MODEL=medium \
  turboocr:cuda12-sm89
```

```
[entrypoint] GPU            NVIDIA L40S (sm89)
[entrypoint] CUDA           driver 12.4 · runtime 12.4 (build-time)
[entrypoint] TensorRT       10.16.0.72 (build-time)
[entrypoint] OCR_MODEL      medium
[entrypoint] engine cache   /models/engines/sm89-rt12.4-drv12.4-trt10.16
[entrypoint] engines found  7
```

---

## Read this first: what is runtime, what is build-time

This is the part that bites people. A cached engine is only valid on a host
whose **GPU architecture, CUDA driver, CUDA runtime and TensorRT version** all
match the machine that built it — all four are folded into the cache key in
`src/engine/trt/trt_engine_cache.cpp`. Only some of them can be chosen when the
container starts.

| Knob | When | Why |
|---|---|---|
| **Model tier** (`OCR_MODEL`) | **runtime** | selects which ONNX is loaded; engines for each tier hash differently |
| **GPU device** (`GPU_DEVICE`) | **runtime** | sets `CUDA_VISIBLE_DEVICES` before any CUDA call |
| **GPU architecture** | *detected* | read from the selected card via `nvidia-smi --query-gpu=compute_cap` |
| **CUDA driver version** | *detected* | a property of the host, not the container |
| **CUDA runtime version** | **build** | compiled into the binary — `--build-arg CUDA_RUNTIME` + matching base image |
| **TensorRT version** | **build** | linked into the binary — `--build-arg TRT_VERSION` |
| **`CMAKE_CUDA_ARCHITECTURES`** | **build** | `--build-arg CUDA_ARCH` |

**CUDA and TensorRT versions are deliberately not runtime env vars.** Setting
one at `docker run` would rename the cache directory without changing a single
byte of what the binary links — which is precisely how you mount an
incompatible cache and then wait hours for a "silent" rebuild. They are stamped
into the image as `TURBO_CUDA_RUNTIME` / `TURBO_TRT_VERSION` for the entrypoint
to report and use, and as OCI labels so you can inspect an image without
running it:

```bash
docker inspect turboocr:cuda12-sm89 --format '{{json .Config.Labels}}'
```

---

## Volume layout

```
/models/
└── engines/
    ├── sm89-rt12.4-drv12.4-trt10.16/     ← L40S / L4 / RTX 40xx on a 550.x driver
    │   ├── VARIANT                        ← manifest, written on first use
    │   ├── det_18046345648738166925.trt   ← medium detection
    │   ├── rec_3758794541574335194.trt    ← medium recognition
    │   ├── det_4878798286114478229.trt    ← small detection
    │   ├── rec_1122236631556611829.trt    ← small recognition
    │   ├── cls_8027655355573618557.trt    ← shared across tiers
    │   ├── layout_10111526822910511266.trt
    │   └── doc_ori_7722513224905535437.trt
    └── sm90-rt12.4-drv12.4-trt10.16/     ← H100, built separately
```

**The model tier is not in the path, on purpose.** `tiny`, `small` and `medium`
engines hash differently, so all three live side by side in one variant
directory and `OCR_MODEL` alone picks between them. `cls`, `layout` and
`doc_ori` are tier-independent and shared by all of them — which is why the
`small` bundle adds only ~70 MB on top of `medium`.

One volume can therefore serve a whole heterogeneous fleet: each node writes
and reads only its own variant directory.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | root of the volume |
| `OCR_MODEL` | `medium` | `tiny` · `small` · `medium`, plus `arabic`, `eslav`, `korean`, `thai`, `greek` |
| `GPU_DEVICE` | — | physical GPU index on a multi-GPU host → `CUDA_VISIBLE_DEVICES` |
| `GPU_ARCH` | *detected* | override the detected `sm_XX` (e.g. `89`); rarely needed |
| `DRIVER_CUDA` | *detected* | override the detected driver CUDA version |
| `TURBO_ENGINE_VARIANT` | *composed* | override the whole variant directory name |
| `TRT_OPT_LEVEL` | `5` | part of the cache key; `3` builds far faster, marginally slower engines |
| `WARM_ONLY` | `0` | `1` = build engines into the volume, then exit |
| `PADDLEX_API` | `0` / `1` | `1` in the PaddleX image: serve the compat API on 8080 |
| `PADDLEX_WORKERS` | `4` | uvicorn workers for the compat API |
| `PIPELINE_POOL_SIZE` | *auto* | pipeline replicas; auto-sizes from VRAM and caps at 5 |

### Build arguments

| Arg | Default | Notes |
|---|---|---|
| `CUDA_ARCH` | `89` | `89` Ada · `90` Hopper/H100 · `120` Blackwell · `86` Ampere |
| `CUDA_RUNTIME` | `12.4` | must match `CUDA_IMAGE`; lands in the cache key |
| `CUDA_IMAGE` / `CUDA_RUNTIME_IMAGE` | `nvidia/cuda:12.4.1-*-ubuntu22.04` | change together with `CUDA_RUNTIME` |
| `TRT_VERSION` | `10.16.0.72` | lands in the cache key |
| `TRT_CUDA` | `12.9` | CUDA flavour of the TensorRT tarball |
| `ORT_VERSION` | `1.22.0` | CUDA-12 GPU build |

`CUDA_ARCH` also selects which TensorRT builder-resource library is copied into
the runtime stage (`libnvinfer_builder_resource_sm${CUDA_ARCH}.so`), so an
image built for one arch stays lean.

---

## Recipes

### Pre-bake a cache (recommended)

Do the multi-hour build once, in a job, instead of on a pod that is supposed to
become ready. `WARM_ONLY=1` builds engines into the volume and exits.

```bash
for TIER in medium small; do
  docker run --rm --gpus all \
    -v turboocr-models:/models \
    -e WARM_ONLY=1 -e OCR_MODEL=$TIER -e TRT_OPT_LEVEL=3 \
    turboocr:cuda12-sm89
done
```

Then serve read-only, so a cache miss fails loudly instead of quietly
rebuilding for hours:

```bash
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models:ro \
  -e OCR_MODEL=medium \
  turboocr:cuda12-sm89
```

### Seed from the published releases

Pre-built Ada engines are published for both tiers:

```bash
VAR=/models/engines/sm89-rt12.4-drv12.4-trt10.16
mkdir -p "$VAR" && cd "$VAR"
BASE=https://github.com/nilavghosh/TurboOCR/releases/download
curl -fsSL "$BASE/engines-l40s-sm89-trt10.16-cuda12.4/turboocr-engines-l40s-sm89-trt10.16-cuda12.4.tar.gz" | tar xz
curl -fsSL "$BASE/engines-small-l40s-sm89-trt10.16-cuda12.4/turboocr-engines-small-l40s-sm89-trt10.16-cuda12.4.tar.gz" | tar xz
```

Both tiers now coexist; switch between them with `OCR_MODEL` and no rebuild.
These are valid only on **sm_89 + driver 550.x + CUDA runtime 12.4 + TRT
10.16** — see [DEPLOY.md](../benchmarks/l40s-ppocrv6-medium/DEPLOY.md).

### Pin a specific GPU

```bash
docker run --gpus all -e GPU_DEVICE=1 -v turboocr-models:/models turboocr:cuda12-sm89
```

Set before any CUDA call, so architecture detection reads the card that will
actually serve.

### Build for an H100

```bash
docker build -f benchmarks/l40s-ppocrv6-medium/docker/Dockerfile.cuda12 \
  --build-arg CUDA_ARCH=90 -t turboocr:cuda12-sm90 .
```

It writes to `/models/engines/sm90-.../` and never touches the Ada engines in
the same volume. Note that the first run rebuilds from ONNX — engines do not
transfer across architectures — so warm it with `TRT_OPT_LEVEL=3` first.

### PaddleX-compatible API

Same flags; the compat API replaces the native one on 8080.

```bash
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models -e OCR_MODEL=medium \
  turboocr:paddlex-sm89
```

Differences from a real PaddleX deployment:
[PADDLEX_COMPAT.md](../compat/paddlex/PADDLEX_COMPAT.md). Measured overhead:
[0% on medium](../benchmarks/l40s-ppocrv6-medium/LOAD40.md),
[2.1% on small](../benchmarks/l40s-ppocrv6-small/LOAD40.md).

---

## Verifying

The entrypoint prints what it resolved and how many engines it found. The
decisive check is still the server's own log:

```bash
docker logs <container> 2>&1 | grep -c "Building TRT engine"
```

**Zero is what you want.** Anything else means a cache miss and a rebuild is
already underway.

The entrypoint also writes a `VARIANT` manifest into each directory on first
use and warns when a mounted cache does not match the running host:

```
[entrypoint] WARNING: cache manifest mismatch -- engines here will be ignored and rebuilt
[entrypoint]          found:    arch=sm89 driver_cuda=12.4 runtime_cuda=12.4 tensorrt=10.16.0.72
[entrypoint]          expected: arch=sm90 driver_cuda=12.4 runtime_cuda=12.4 tensorrt=10.16.0.72
```

That warning is the whole point of the layout: without it, a mismatch is
invisible until you notice startup took three hours.

---

## Things that will surprise you

**A host driver upgrade invalidates every engine.** `cudaDriverGetVersion()` is
in the cache key, so 550.x → 580.x changes the variant name and every engine
rebuilds. Warm the new variant *before* rolling drivers, not after. The
directory naming makes this survivable: both variants coexist, so you can
pre-build the new one while the old one still serves.

**The ONNX files must stay in the image.** The cache key is derived from each
ONNX file's size and mtime, so they are read even on a cache hit. They ship in
the image at `/app/models` with mtimes preserved from the release. Do not move
them into the volume unless you copy with `cp -a` / `tar -p` — a plain copy or
a `git checkout` rewrites mtimes and silently invalidates the whole cache.

**Read-only is a feature.** Mounting `/models:ro` converts a silent multi-hour
rebuild into an immediate, obvious failure. Use it everywhere except the
warming job.

**`PIPELINE_POOL_SIZE` caps at 5 regardless of VRAM.** Auto-sizing tops out at
5 pipelines because throughput plateaus there — an 80 GB H100 gets the same
pool as a 46 GB L40S, and filling VRAM buys nothing. See `pool_sizing.h`.
