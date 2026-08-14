# Deploying with a pre-built TensorRT engine cache

First start builds five TensorRT engines from ONNX. On this L40S that took
**~3 hours** at the default `TRT_OPT_LEVEL=5`. Mounting a pre-built cache
reduces startup to **~7 seconds**.

This document covers where the engines live, exactly when they can be reused,
and how to attach them to a container.

---

## Where the engines are

Default location is `$HOME/.cache/turbo-ocr` (overridable with
`TRT_ENGINE_CACHE`). From this benchmark run:

| File | Size | Stage |
|---|---:|---|
| `det_18046345648738166925.trt` | 33 MB | PP-OCRv6 medium detection |
| `rec_3758794541574335194.trt` | 128 MB | PP-OCRv6 medium recognition |
| `layout_10111526822910511266.trt` | 71 MB | PP-DocLayoutV3 |
| `doc_ori_7722513224905535437.trt` | 5.2 MB | page orientation |
| `cls_8027655355573618557.trt` | 1.4 MB | line orientation |

**229 MB total.** Packaged on the benchmark host as:

```
/workspace/turboocr-deploy/turboocr-engines-l40s-sm89-trt10.16-cuda12.4.tar.gz   (165 MB)
/workspace/turboocr-deploy/engines.sha256
```

These are **not committed to git** — the recognizer engine alone is 128 MB,
over GitHub's 100 MB per-file limit. Distribute them as a release asset, an
object-store object, or a pre-seeded Docker volume.

---

## When a cached engine is reused

This is the part that decides whether mounting a cache actually helps. The
filename *is* the cache key — a 64-bit hash of a string assembled in
`src/engine/trt/trt_engine_cache.cpp`. Any component that differs produces a
different filename, the file is not found, and the engine is silently rebuilt
from ONNX.

The key for the detection engine above, verified by reproducing the hash
exactly:

```
v20260615:det:models/det.onnx:62032837:-4655052668000000000:10.16.0:sm8.9:drv12040:rt12040:dms1280:dob4:opt5
  -> 18046345648738166925  ->  det_18046345648738166925.trt
```

Component by component:

| Component | This build | Reused only when |
|---|---|---|
| `v20260615` | profile version constant | the source's `kProfileVersion` is unchanged |
| `det` | stage name | — |
| `models/det.onnx` | **relative** ONNX path | the server runs with the same CWD and model layout |
| `62032837` | ONNX file size | same model release |
| `-4655052668000000000` | ONNX mtime (`file_clock` ns) | **the ONNX file's mtime is preserved** |
| `10.16.0` | TensorRT major.minor.patch | same TensorRT version |
| `sm8.9` | GPU compute capability | same GPU architecture (Ada: L40S, L4, RTX 40xx) |
| `drv12040` | `cudaDriverGetVersion()` | same **host NVIDIA driver** branch (550.x → 12040) |
| `rt12040` | `cudaRuntimeGetVersion()` | binary built against the same CUDA runtime (12.4) |
| `dms1280:dob4` | det max-side, opt batch | `DET_MAX_SIDE*` / `DET_OPT_BATCH` unchanged |
| `opt5` | `TRT_OPT_LEVEL` | same optimization level |

### Consequences worth planning around

**A host driver upgrade invalidates every engine.** `drv` comes from
`cudaDriverGetVersion()`, so moving from a 550.x driver to a 580.x driver
changes the key and triggers a full multi-hour rebuild. Rebuild the cache
*before* rolling the driver, not after.

**The official `ghcr.io/aiptimizer/turboocr` image cannot reuse these
engines.** It is built on the nvcr TensorRT 26.03 base (CUDA 13), so `rt`
would be `130xx` rather than `12040`. On a 550.x driver that image will not
run at all — CUDA 13 requires driver 580+. That is why the image in
`docker/Dockerfile.cuda12` here pins the CUDA 12 stack.

**The ONNX files must ship in the image even when engines are cached.**
`get_cached_engine_path()` calls `fs::file_size()` and `fs::last_write_time()`
on the ONNX path to compute the key, so the files have to exist and carry
their original mtimes. `scripts/fetch_release_models.sh` uses `wget`, which
preserves the release's `Last-Modified` timestamps, so fetching the models
during the build yields the same mtimes on every machine — this is what makes
a cache portable at all. `COPY` also preserves mtimes; `git checkout` does
**not**, so never source the ONNX files from a git working tree.

**Engines are GPU-architecture specific.** An sm_89 cache is useless on an
sm_90 (H100) or sm_120 (RTX 50) host.

---

## Building the image

```bash
docker build -f benchmarks/l40s-ppocrv6-medium/docker/Dockerfile.cuda12 \
             -t turboocr:cuda12 .
```

Build arguments:

| Arg | Default | Notes |
|---|---|---|
| `CUDA_IMAGE` | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | sets `rt` in the cache key |
| `TRT_VERSION` | `10.16.0.72` | sets the TRT component of the key |
| `TRT_CUDA` | `12.9` | CUDA flavor of the TensorRT tarball |
| `ORT_VERSION` | `1.22.0` | CUDA-12 GPU build |
| `CUDA_ARCH` | `89` | Ada; change for other GPUs |

To match a cache built against a different CUDA runtime, change `CUDA_IMAGE`
and `CUDA_RUNTIME_IMAGE` together — the `rt` value follows the runtime the
binary is compiled against.

> **Not verified.** Docker was unavailable on the benchmark host, so this
> Dockerfile was written from the verified native build steps but has never
> been built. Treat the first build as something to babysit. The native build
> it mirrors is fully verified and is documented in `README.md`.

---

## Running with a mounted cache

```bash
# one-time: unpack the engines somewhere the daemon can read
mkdir -p /opt/turboocr-engines
tar xzf turboocr-engines-l40s-sm89-trt10.16-cuda12.4.tar.gz -C /opt/turboocr-engines
sha256sum -c engines.sha256   # run from inside that directory

docker run --gpus all -p 8080:8080 -p 50051:50051 \
  -v /opt/turboocr-engines:/engines:ro \
  -e TRT_ENGINE_CACHE=/engines \
  -e OCR_MODEL=medium \
  turboocr:cuda12
```

Read-only is fine and is a useful guard: on a cache miss the server cannot
write a new engine, so it fails loudly instead of quietly burning hours
rebuilding. Drop `:ro` if you would rather it self-heal.

Using a named volume instead, seeded once:

```bash
docker volume create turboocr-engines
docker run --rm -v turboocr-engines:/engines \
  -v /opt/turboocr-engines:/seed:ro alpine \
  sh -c 'cp /seed/*.trt /engines/'
```

Then mount `-v turboocr-engines:/engines`.

### Verify the cache was actually used

```bash
docker logs <container> 2>&1 | grep -c "Building TRT engine"
```

**Zero is what you want.** Any hit means a key mismatch and a rebuild is
underway — compare the expected filenames against `/engines` and work through
the table above to find which component differs.

Readiness:

```bash
curl -fsS http://localhost:8080/health/ready     # "ok" once warm
curl -s   http://localhost:8080/capabilities
curl -X POST http://localhost:8080/ocr/raw \
     --data-binary @tests/fixtures/images/png/receipt.png \
     -H 'Content-Type: image/png'
```

With a warm cache the server is ready in about 7 seconds; `/health/ready`
returns 503 until then.

---

## Rebuilding the cache from scratch

On a new GPU architecture or after a driver upgrade, generate a fresh cache by
starting the server once and waiting:

```bash
docker run --gpus all -v /opt/turboocr-engines-new:/engines \
  -e TRT_ENGINE_CACHE=/engines -e OCR_MODEL=medium turboocr:cuda12
```

Budget several hours on non-Blackwell hardware. `TRT_OPT_LEVEL=3` cuts build
time substantially — it reduced the detection engine from ~10 min to ~3 min
here — at the cost of a small steady-state throughput regression. Note that
the level is itself part of the cache key, so engines built at level 3 and
level 5 coexist as separate files.

Run this once per architecture, archive the result, and mount it everywhere
else.
