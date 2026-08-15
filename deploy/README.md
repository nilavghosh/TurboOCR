# TurboOCR deployment guide

End-to-end: choose a configuration, build an image, warm the engine cache, run
it, and operate it. Everything here was measured on an NVIDIA L40S — see
[benchmarks](../benchmarks/) for the raw numbers behind each claim.

1. [Quick start](#quick-start)
2. [Choosing a configuration](#choosing-a-configuration)
3. [Build-time vs runtime](#build-time-vs-runtime) ← read before anything else
4. [Building the image](#building-the-image)
5. [Warming the engine cache](#warming-the-engine-cache)
6. [Running](#running)
7. [Verifying](#verifying)
8. [Operating](#operating)
9. [Troubleshooting](#troubleshooting)
10. [Reference](#reference)

---

## Quick start

Ada GPU (L40S / L4 / RTX 40xx) on a 550.x driver, using the published engines:

```bash
# 1 · seed the volume with pre-built engines
docker volume create turboocr-models
docker run --rm -v turboocr-models:/models -w /models alpine sh -c '
  mkdir -p engines/sm89-rt12.4-drv12.4-trt10.16 &&
  cd engines/sm89-rt12.4-drv12.4-trt10.16 &&
  wget -qO- https://github.com/nilavghosh/TurboOCR/releases/download/engines-l40s-sm89-trt10.16-cuda12.4/turboocr-engines-l40s-sm89-trt10.16-cuda12.4.tar.gz | tar xz'

# 2 · build
docker build -f benchmarks/l40s-ppocrv6-medium/docker/Dockerfile.cuda12 \
  --build-arg CUDA_ARCH=89 -t turboocr:cuda12-sm89 .

# 3 · run — ready in ~7 s instead of ~3 h
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models -e OCR_MODEL=medium \
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

Any other GPU or driver: skip step 1 and [warm your own
cache](#warming-the-engine-cache) — engines are not portable across
architectures or driver branches.

---

## Choosing a configuration

### Model tier

Measured on one L40S, FUNSD, `OCR_MODEL` the only variable:

| Tier | Word F1 | Throughput | p50 (c=1) | 25-user p50 | VRAM |
|---|---:|---:|---:|---:|---:|
| `medium` | **92.34%** | 48.5 img/s | 34.6 ms | 670 ms | 39.3 GB |
| `small` | 90.81% | **143.4 img/s** | **29.0 ms** | **210 ms** | 27.2 GB |
| `tiny` | ~84.6%* | ~6× medium* | — | — | — |

<sub>*`tiny` not measured here; figures are the project's own RTX 5090 table.</sub>

**`small` is 3× the throughput for 1.5 points of F1** and is the right default
for most serving. Choose `medium` when accuracy dominates and you can afford a
third of the throughput.

One caveat that matters if you are optimising single-request latency: at
concurrency 1 the tier change buys only **1.19×** (34.6 → 29.0 ms), because most
of one request is fixed cost the model never touches. The 3× appears only under
load. Size from loaded numbers, not idle ones.

### GPU

The workload is **GPU-bound, not VRAM-bound** — 93–99% GPU utilisation against
under 6% CPU. `PIPELINE_POOL_SIZE` auto-sizes from VRAM but **caps at 5**,
because throughput plateaus there (`include/turbo_ocr/server/bootstrap/pool_sizing.h`).
An 80 GB H100 gets the same pool as a 46 GB L40S; buying VRAM buys nothing here.
VRAM still sets a *floor*, though — see
[VRAM and pipeline pool size](#vram-and-pipeline-pool-size) before deploying to
anything smaller than ~40 GB.

Rough capacity, `small` tier: **~143 img/s per L40S.** Scale out with more
cards rather than up to a bigger one — and note these numbers were taken on a
thermally throttled chassis, so a well-cooled host does better.

### VRAM and pipeline pool size

Measured on the L40S, both tiers, by varying `PIPELINE_POOL_SIZE`. Usage is
cleanly linear in the pool size:

| Pool | `medium` | `small` |
|---:|---:|---:|
| 1 | 8,271 MiB | 5,775 MiB |
| 2 | 16,009 MiB | 11,015 MiB |
| 3 | 23,779 MiB | 16,259 MiB |
| 5 *(default)* | **39,283 MiB** | **26,773 MiB** |

Which fits:

```
medium ≈  518 MiB + 7,753 MiB × pool
small  ≈  526 MiB + 5,250 MiB × pool
```

Use it to check a card before deploying to it:

| GPU | VRAM | Max pool, `medium` | Max pool, `small` |
|---|---:|---:|---:|
| L40S / A6000 Ada | 46 GB | 5 *(default)* | 5 *(default)* |
| A100 | 40 GB | 5 | 5 |
| **L4** | **24 GB** | **2** | **4** |
| A10 / A10G / RTX 4090 | 24 GB | 2 | 4 |
| RTX 4000 Ada | 20 GB | 2 | 3 |
| A2 / T4 | 16 GB | 1 | 2 |

This table is about **VRAM only**. Engine *compatibility* is a separate axis:
only Ada cards (`sm_89` — L40S, L4, RTX 40xx, RTX 4000 Ada) share engines with
the published Ada bundles. A10/A10G are Ampere (`sm_86`) and T4 is Turing
(`sm_75`); both need their own variant built. See
[Build-time vs runtime](#build-time-vs-runtime).

### Running on an L4 (or any 24 GB card)

**The engines transfer without a rebuild.** An L4 is AD104 — compute capability
**8.9**, the same `sm_89` as an L40S — so the cache key matches and the
entrypoint resolves the same variant directory. The one thing to confirm is the
driver: `cudaDriverGetVersion()` is part of the key, so the L4 host must report
the same CUDA version (a 550.x driver reports 12.4). On a different driver
branch it becomes a different variant and rebuilds.

**But you must set the pool size by hand.** Neither tier fits at the default
pool of 5 on 24 GB:

```bash
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models:ro \
  -e OCR_MODEL=small -e PIPELINE_POOL_SIZE=4 \
  turboocr:cuda12-sm89
```

Use `PIPELINE_POOL_SIZE=2` for `medium`. Leave a pipeline of headroom if you
also enable tables or formulas, which load further engines.

> **Auto-sizing will not protect you here.** `compute_pipeline_pool_size()`
> sees 24 GB ≥ 14 GB and selects 5. Its safety check then estimates **2 GiB per
> pipeline** — roughly **4× too low for `medium`** — concludes that 11 would
> fit, and leaves the pool at 5. The card then OOMs during warmup with a fatal
> `cuda_ptr.h - out of memory`. On any card below ~40 GB, set
> `PIPELINE_POOL_SIZE` explicitly rather than trusting the default.

Set expectations on throughput separately from VRAM: an L4 is a 72 W part
against the L40S's 350 W with roughly a third of the memory bandwidth, so
expect well under a third of the ~143 img/s measured for `small` — and a
smaller pool compounds that. For throughput per pound, the L40S is the better
card here.

### API

| | Native | PaddleX-compatible |
|---|---|---|
| Image | `Dockerfile.cuda12` | `Dockerfile.paddlex` |
| Endpoint | `/ocr/raw`, `/ocr/pdf`, … | `POST /ocr`, `GET /health` |
| Overhead | — | **0% on medium**, 2.1% on small |

The compat layer is effectively free — on `medium` the GPU is already 99.7%
utilised, so the adapter's CPU work hides entirely. Differences from a real
PaddleX deployment: [PADDLEX_COMPAT.md](../compat/paddlex/PADDLEX_COMPAT.md).

---

## Build-time vs runtime

**This is the section that prevents multi-hour mistakes.**

A cached TensorRT engine is valid only on a host whose **GPU architecture, CUDA
driver, CUDA runtime and TensorRT version** all match the machine that built it.
All four are folded into the cache key in
`src/engine/trt/trt_engine_cache.cpp`. When any differs, the engine file is not
found and TensorRT **rebuilds from ONNX** — hours, with no error.

| Knob | When | How |
|---|---|---|
| Model tier | **runtime** | `OCR_MODEL` |
| GPU device | **runtime** | `GPU_DEVICE` → `CUDA_VISIBLE_DEVICES` |
| Optimisation level | **runtime** | `TRT_OPT_LEVEL` (also in the cache key) |
| GPU architecture | *detected* | `nvidia-smi --query-gpu=compute_cap`; `GPU_ARCH` overrides |
| CUDA driver version | *detected* | a host property |
| CUDA runtime version | **build** | `--build-arg CUDA_RUNTIME` + matching base image |
| TensorRT version | **build** | `--build-arg TRT_VERSION` |
| `CMAKE_CUDA_ARCHITECTURES` | **build** | `--build-arg CUDA_ARCH` |

**CUDA and TensorRT versions are deliberately not runtime env vars.** Setting
one at `docker run` would rename the cache directory without changing a byte of
what the binary links — exactly how an incompatible cache gets mounted and then
silently rebuilt. They are stamped into the image as `TURBO_CUDA_RUNTIME` /
`TURBO_TRT_VERSION` and as OCI labels, inspectable without running anything:

```bash
docker inspect turboocr:cuda12-sm89 --format '{{json .Config.Labels}}'
```

### Volume layout

```
/models/
└── engines/
    ├── sm89-rt12.4-drv12.4-trt10.16/     ← L40S / L4 / RTX 40xx, driver 550.x
    │   ├── VARIANT                        ← manifest, written on first use
    │   ├── det_18046345648738166925.trt   ← medium detection
    │   ├── rec_3758794541574335194.trt    ← medium recognition
    │   ├── det_4878798286114478229.trt    ← small detection
    │   ├── rec_1122236631556611829.trt    ← small recognition
    │   ├── cls_8027655355573618557.trt    ← tier-independent
    │   ├── layout_10111526822910511266.trt
    │   └── doc_ori_7722513224905535437.trt
    └── sm90-rt12.4-drv12.4-trt10.16/     ← H100, built separately
```

**The tier is not in the path, deliberately.** `tiny`/`small`/`medium` engines
hash differently, so they coexist in one variant directory and `OCR_MODEL`
alone picks between them with no rebuild. `cls`, `layout` and `doc_ori` are
tier-independent and shared — which is why adding `small` on top of `medium`
costs only ~70 MB. One volume can serve a heterogeneous fleet; each node reads
and writes only its own variant.

---

## Building the image

```bash
docker build -f benchmarks/l40s-ppocrv6-medium/docker/Dockerfile.cuda12 \
  --build-arg CUDA_ARCH=90 \
  --build-arg CUDA_RUNTIME=12.4 \
  --build-arg TRT_VERSION=10.16.0.72 \
  --build-arg TRT_CUDA=12.9 \
  -t turboocr:cuda12-sm90 .
```

| Arg | Default | Notes |
|---|---|---|
| `CUDA_ARCH` | `90` | `86` Ampere · `89` Ada/L40S · `90` Hopper/H100 · `120` Blackwell |
| `CUDA_RUNTIME` | `12.4` | must match `CUDA_IMAGE`; lands in the cache key |
| `CUDA_IMAGE` / `CUDA_RUNTIME_IMAGE` | `nvidia/cuda:12.4.1-*-ubuntu22.04` | change together with `CUDA_RUNTIME` |
| `TRT_VERSION` | `10.16.0.72` | lands in the cache key |
| `TRT_CUDA` | `12.9` | CUDA flavour of the TensorRT tarball |
| `ORT_VERSION` | `1.22.0` | CUDA-12 GPU build |

`CUDA_ARCH` also selects which TensorRT builder-resource library is copied into
the runtime stage, keeping the image lean. It is therefore not merely a codegen
hint: an image built for one architecture cannot build engines on another, and a
cold cache on a mismatched card fails with an opaque "Failed to build engine".
The default is `90` because that is the deployment target; the `l40s-` directory
name reflects where the medium-tier benchmarks were measured, not what the image
defaults to. For an L40S pass `--build-arg CUDA_ARCH=89`.

**Why a CUDA 12 image at all:** the upstream `ghcr.io/aiptimizer/turboocr` is
built on CUDA 13 and needs driver 580+. On a 550.x host it will not run, and
its engines are not interchangeable with these (`rt` differs). These Dockerfiles
exist for hosts on a 12.x driver.

> **Not yet built.** Docker was unavailable on the benchmark host, so these
> Dockerfiles are written from a fully verified native build but have never been
> executed. Babysit the first build; the likeliest snags are runtime-stage apt
> package names on Ubuntu 22.04. The native build they mirror is documented in
> [the medium README](../benchmarks/l40s-ppocrv6-medium/README.md#building-without-docker).

---

## Warming the engine cache

First start compiles five TensorRT engines from ONNX. Measured on the L40S at
the default `TRT_OPT_LEVEL=5`:

| Engine | medium | small |
|---|---:|---:|
| `det` | ~10 min | ~7 min |
| **`rec`** | **~2 h 15 m** | **~1 h 55 m** |
| `layout` | ~25 min | *shared* |
| `cls`, `doc_ori` | ~6 min | *shared* |
| **Total** | **~3 h** | **~2 h** |

The recognizer dominates: one optimisation profile spans width 48→4000 and
batch 1→32, so the level-5 autotuner searches an enormous space. Build time
tracks graph structure, not weight size — `small` is only marginally faster
despite being a third of the bytes.

**Do this once, in a job.** `WARM_ONLY=1` builds into the volume and exits
without serving:

```bash
for TIER in medium small; do
  docker run --rm --gpus all \
    -v turboocr-models:/models \
    -e WARM_ONLY=1 -e OCR_MODEL=$TIER -e TRT_OPT_LEVEL=3 \
    turboocr:cuda12-sm89
done
```

`TRT_OPT_LEVEL=3` cuts build time substantially (detection went ~10 min → ~3 min)
for a small steady-state cost. It is part of the cache key, so level-3 and
level-5 engines coexist as separate files — pick one and stay on it.

Archive the result and reuse it across every identical host:

```bash
tar czf engines-sm89-rt12.4-drv12.4-trt10.16.tar.gz -C /var/lib/docker/volumes/turboocr-models/_data/engines/sm89-rt12.4-drv12.4-trt10.16 .
```

---

## Running

### Native API

```bash
docker run --gpus all -p 8080:8080 -p 50051:50051 \
  -v turboocr-models:/models:ro \
  -e OCR_MODEL=small \
  turboocr:cuda12-sm89
```

**Mount read-only in production.** A cache miss then fails immediately instead
of quietly rebuilding for hours. Use read-write only for the warming job.

### PaddleX-compatible API

```bash
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models:ro \
  -e OCR_MODEL=small \
  turboocr:paddlex-sm90
```

The adapter owns 8080 (PaddleX's default) and the backend binds 8081 privately,
so an existing client moves with a hostname change:

```bash
curl -X POST http://localhost:8080/ocr -H 'Content-Type: application/json' \
  -d "{\"file\": \"$(base64 -w0 page.png)\", \"fileType\": 1}"
```

Set `PADDLEX_STRICT_PARAMS=1` while migrating: it turns silently-ignored
parameters into a 422 naming them, so you find out which callers depend on
knobs this backend cannot honour.

### Pinning a GPU

```bash
docker run --gpus all -e GPU_DEVICE=1 -v turboocr-models:/models turboocr:cuda12-sm89
```

Applied before any CUDA call, so architecture detection reads the card that
will actually serve.

---

## Verifying

```bash
docker logs <container> 2>&1 | grep -c "Building TRT engine"
```

**Zero is what you want.** Anything else means a cache miss and a rebuild is
already underway.

```bash
curl -fsS http://localhost:8080/health/ready      # "ok"; 503 until warm
curl -s   http://localhost:8080/capabilities      # which stages loaded
curl -X POST http://localhost:8080/ocr/raw \
     --data-binary @tests/fixtures/images/png/receipt.png \
     -H 'Content-Type: image/png'
```

With a warm cache the server is ready in about **7 seconds** (measured: 24 s
including container start when switching tiers).

Prometheus metrics are on `/metrics`, including
`turbo_ocr_pipeline_pool_size`.

---

## Operating

**A host driver upgrade invalidates every engine.** `cudaDriverGetVersion()` is
in the cache key, so 550.x → 580.x renames the variant and everything rebuilds.
Warm the new variant *before* rolling drivers. The layout makes this survivable:
both variants coexist, so the new one can be built while the old one still
serves.

**Keep the ONNX files in the image.** The cache key is derived from each ONNX
file's size and mtime, so they are read even on a cache hit. They ship at
`/app/models` with mtimes preserved from the release. Moving them into the
volume requires `cp -a` / `tar -p`; a plain copy or a `git checkout` rewrites
mtimes and invalidates the entire cache.

**Scaling.** Throughput saturates at concurrency 4 and holds flat to 40 with
zero errors, so a modest client concurrency is enough to keep a card busy.
`PIPELINE_POOL_SIZE` caps at 5 regardless of VRAM. Scale out with more
replicas/GPUs, not a larger pool.

**Watch thermals.** On the benchmark chassis the L40S hit 88 °C and dropped
from a 2520 MHz boost to as low as 555 MHz, costing up to 23% of sustained
throughput on `medium`. If measured throughput is below expectation, check
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active`
before suspecting the software.

**PDF inputs.** `mode=geometric` extracts an existing text layer at ~280
pages/s against ~11 for full OCR — 25×. For born-digital PDFs, `mode=auto`
matters far more to throughput than the model tier does.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Startup takes hours; log shows `Building TRT engine` | cache miss | compare the entrypoint's variant against the directories in `/models/engines`; check the `VARIANT` manifest warning |
| `WARNING: cache manifest mismatch` | engines built on a different host | warm this variant, or fix the mismatched component |
| `ERROR: … is missing and /models is not writable` | read-only mount, no engines for this variant | warm it read-write first |
| `ERROR: nvidia-smi not found` | container started without GPU | add `--gpus all` |
| Container exits at startup, `GLIBC_2.38 not found` | vendored `fastpdf2png` vs the base image's glibc | already handled in these Dockerfiles (rebuilt from source) |
| Fatal `cuda_ptr.h - out of memory` at startup | pool too large for the card | set `PIPELINE_POOL_SIZE` — see [VRAM and pipeline pool size](#vram-and-pipeline-pool-size); auto-sizing under-estimates by ~4× on `medium` |
| `/health/ready` returns 503 forever | engines still building, or a fatal init error | check logs; a CUDA OOM here usually means another process holds the GPU |
| PaddleX client gets `null` images | `visualize` is unsupported by design | see [PADDLEX_COMPAT.md](../compat/paddlex/PADDLEX_COMPAT.md) |
| Throughput well below the tables above | thermal throttling | check SM clock and throttle reasons |

---

## Reference

### Runtime environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | volume root |
| `OCR_MODEL` | `medium` | `tiny`·`small`·`medium`, plus `arabic`, `eslav`, `korean`, `thai`, `greek` |
| `GPU_DEVICE` | — | physical GPU index → `CUDA_VISIBLE_DEVICES` |
| `GPU_ARCH` | *detected* | override detected `sm_XX` |
| `DRIVER_CUDA` | *detected* | override detected driver CUDA version |
| `TURBO_ENGINE_VARIANT` | *composed* | override the whole variant directory name |
| `TRT_OPT_LEVEL` | `5` | in the cache key; `3` builds much faster |
| `WARM_ONLY` | `0` | `1` = build engines and exit |
| `PIPELINE_POOL_SIZE` | *auto* | pipeline replicas; auto-sizes from VRAM and caps at 5. **Set explicitly below ~40 GB** — see [VRAM and pipeline pool size](#vram-and-pipeline-pool-size) |
| `DET_MAX_SIDE_LIMIT` | `1280` | detection resolution; **in the cache key** |
| `DISABLE_LAYOUT` | `0` | skip loading the layout model |
| `TABLE_BACKEND` / `FORMULA_BACKEND` | — | opt-in table → HTML, formula → LaTeX |
| `PADDLEX_API` | image default | `1` serves the compat API |
| `PADDLEX_WORKERS` | `4` | uvicorn workers |
| `PADDLEX_STRICT_PARAMS` | `0` | 422 on unsupported params instead of ignoring |
| `LOG_FORMAT` / `LOG_LEVEL` | `json` / `info` | |

### Further reading

| Document | Covers |
|---|---|
| [Engine cache internals](../benchmarks/l40s-ppocrv6-medium/DEPLOY.md) | the cache key derived component by component |
| [PaddleX compatibility](../compat/paddlex/PADDLEX_COMPAT.md) | field-by-field API differences |
| [medium benchmark](../benchmarks/l40s-ppocrv6-medium/README.md) | accuracy, throughput, native build steps |
| [small benchmark](../benchmarks/l40s-ppocrv6-small/README.md) | latency-focused tier comparison |
| [40-user load tests](../benchmarks/l40s-ppocrv6-medium/LOAD40.md) | native vs PaddleX, CPU/GPU attribution |
