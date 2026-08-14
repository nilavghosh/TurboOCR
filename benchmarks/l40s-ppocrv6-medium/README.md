# PP-OCRv6 `medium` on an NVIDIA L40S — reproduction run

Independent benchmark of TurboOCR v3.0 built **natively** (no Docker) on an
Ubuntu 22.04 / CUDA 12.4 host with a single NVIDIA L40S, running the
`OCR_MODEL=medium` tier.

Two questions were being answered: do the project's published accuracy and
throughput numbers reproduce on different hardware, and does throughput hold up
under sustained concurrent load.

**Short answer:** accuracy reproduces exactly (92.34% vs 91.9% claimed F1 on
FUNSD). Throughput lands at 56% of the RTX 5090 figure, which is what the
hardware gap predicts. Under a 2-minute sustained load the box loses ~23%
throughput to **GPU thermal throttling** — a chassis-cooling limit, not a
software one.

---

## Test environment

| | |
|---|---|
| GPU | NVIDIA L40S 46 GB (Ada, sm_89), driver 550.144.03 |
| Host | Ubuntu 22.04.5, CUDA 12.4, 112 cores, 503 GB RAM |
| TurboOCR | `ed01c3ea`, GPU build, `OCR_MODEL=medium` |
| Stages | text detection + recognition + line orientation; layout loaded but not requested. Tables/formulas off. |

Reference hardware for the published numbers is an RTX 5090, so throughput here
is expected to be lower; accuracy is hardware-independent and should match.

---

## Building without Docker

The project targets CUDA 13 on Arch Linux. On a CUDA 12.4 / Ubuntu 22.04 host
four substitutions were needed. **No source changes were required** — the tree
compiles clean against the CUDA-12 stack.

| Component | Project pin | Used here | Why |
|---|---|---|---|
| TensorRT | 10.16.0.72 `cuda-13.2` | 10.16.0.72 **`cuda-12.9`** | Same TRT version, CUDA-12 flavor. NVIDIA ships both. |
| ONNX Runtime | 1.27.0 `gpu_cuda13` | 1.22.0 `gpu` (CUDA 12) | The CUDA-13 build will not load against a 12.4 runtime. |
| Compiler | GCC 15 | **GCC 13** (`ubuntu-toolchain-r/test`) | The code uses `std::format`; Ubuntu 22.04's GCC 11 lacks it. |
| `bin/fastpdf2png` | vendored binary | **rebuilt from source** | The shipped binary needs glibc 2.38; Ubuntu 22.04 has 2.35. |

```bash
# TensorRT 10.16 for CUDA 12.9 -> /usr/local/tensorrt
# ONNX Runtime GPU (CUDA 12) -> third_party/onnxruntime/{include,lib}
bash scripts/install_fastpdf2png.sh          # rebuild for the host glibc

CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_DIR=/usr/local/tensorrt \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DFETCH_MODELS=OFF
cmake --build build -j48 --target turboocr-server
```

`CMAKE_CUDA_ARCHITECTURES=89` targets Ada directly instead of relying on PTX
JIT. Models come from `scripts/fetch_release_models.sh`.

### Engine build time is the surprise

First start built all five TensorRT engines in **~3 hours** at the default
`TRT_OPT_LEVEL=5`:

| Engine | Time | Cached size |
|---|---:|---:|
| `det` (medium) | ~10 min | 33 MB |
| **`rec` (medium)** | **~2 h 15 min** | 128 MB |
| `layout` | ~25 min | 71 MB |
| `cls` | <1 min | 1.4 MB |
| `doc_ori` | ~5 min | 5.2 MB |

The recognizer dominates because it uses a single optimization profile spanning
width 48→4000 and batch 1→32 (`src/engine/trt/trt_profiles.cpp`), giving the
level-5 autotuner an enormous tactic search space. The README's guidance of
"~90 s on a 5090 and up to an hour on older GPUs" understates this for an L40S
by roughly 3×. `TRT_OPT_LEVEL=3` cut the detection engine build from ~10 min to
~3 min, so it is worth setting on non-Blackwell cards.

Engines cache to `~/.cache/turbo-ocr` (229 MB total); subsequent starts take
about 7 seconds.

> Note: the cache key includes the CUDA **driver** version, so a host driver
> upgrade invalidates every engine and triggers a full rebuild. Worth knowing
> before patching a production box.

---

## 1. FUNSD — accuracy and throughput

`tests/benchmark/bench_funsd_local.py`, the same benchmark the project quotes.
The repo ships ground truth (`funsd_gt_words.json`) but not images; the FUNSD
test split was fetched from the original source and each of the 50 pages was
verified to align **uniquely and exactly** (word-bag equality against the
dataset's own annotations) with the shipped ground truth, so this is the
project's benchmark rather than an approximation.

| | This run (L40S) | Published (RTX 5090) |
|---|---:|---:|
| Word F1 | **92.34%** | 91.9% |
| Precision / Recall | 91.50% / 93.29% | — |
| Throughput | **48.5 img/s** | 86 img/s |
| Latency | p50 29 ms · p95 46 ms | — |

**Accuracy reproduces and slightly exceeds the published figure.** Throughput is
56% of the 5090 number, consistent with the hardware gap.

For context, from the project's comparison table (all measured on a 5090):
PaddleOCR-VL 5 img/s at 91.6% F1, PaddleOCR PP-OCRv5 Python 6 img/s at 86.6%.
This L40S run is therefore roughly **8–10× faster at higher accuracy than those
engines achieve on a faster GPU** — the TensorRT pipeline is doing real work.

---

## 2. Endpoint throughput matrix

`tests/benchmark/bench_throughput.py --n 200`, fixture `business_letter.png`
(a dense A4-style page). Zero errors across all 6,400 requests.

### Whole-page OCR (img/s)

| Endpoint | c=1 | c=4 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| `POST /ocr` (base64 JSON) | 23.2 | 40.4 | **41.7** | 40.9 |
| `POST /ocr/raw` (binary) | 27.5 | 39.3 | 40.7 | 40.6 |
| `POST /ocr/pixels` (raw pixels) | 31.1 | 40.5 | 39.9 | 39.9 |
| `POST /ocr/batch` | 28.9 | **35.4** | 30.6 | 29.4 |

Saturates at concurrency 4 and stays flat to 32 with no error growth — no
contention collapse. At c=1 `/ocr/pixels` leads (31.1 vs 23.2 for base64 JSON),
showing decode and transport overhead that disappears once the GPU is the
bottleneck.

### PDF (pages/s)

| Mode | c=1 | c=4 | c=16 | c=32 |
|---|---:|---:|---:|---:|
| `mode=ocr` | 11.6 | 11.5 | 11.1 | 10.5 |
| `mode=geometric` | 247.5 | **284.6** | 273.0 | 274.4 |
| `mode=auto` | 254.2 | 281.2 | 283.5 | 254.4 |
| `mode=auto_verified` | 10.9 | 10.8 | 11.0 | 11.1 |

`geometric` is ~25× `ocr`. For digital PDFs carrying a text layer, choosing
`mode=auto` matters far more to throughput than the model tier does.

### Latency at c=1

| Endpoint | p50 | p95 | p99 |
|---|---:|---:|---:|
| `POST /ocr/pixels` | 30.9 ms | 32.1 | 32.8 |
| `POST /ocr/raw` | 34.6 ms | 39.7 | 39.9 |
| `POST /ocr` | 38.5 ms | 46.0 | 46.7 |
| `POST /ocr/batch` (8 images) | 265.8 ms | 299.5 | 308.2 |
| `POST /ocr/pdf?mode=geometric` | 76.3 ms | 98.6 | 110.8 |
| `POST /ocr/pdf?mode=ocr` | 1385 ms | 2164 | 2192 |

---

## 3. Sustained load — 25 concurrent users, 2 minutes

`locustfile.py` in this directory, against `/ocr/raw`. Run twice to confirm
reproducibility.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Requests | 4,284 | 4,079 |
| **Failures** | **0 (0.00%)** | **0 (0.00%)** |
| Throughput (whole-run avg) | 35.9 req/s | 34.2 req/s |
| p50 | 670 ms | 740 ms |
| p95 | 820 ms | 830 ms |
| p99 | 850 ms | 860 ms |
| Max | 937 ms | 930 ms |

Zero failures across 8,363 requests. Every response was validated as genuine OCR
output — the Locust task asserts a `results` array in the body rather than
trusting HTTP 200, so an error payload returned with a 200 cannot inflate the
result.

### Throughput decays during the run

Identical curve in both runs:

| Elapsed | ~10 s | ~30 s | ~60 s | ~90 s | ~120 s |
|---|---:|---:|---:|---:|---:|
| req/s | 41.1 | 40.3 | 37.8 | 33.1 | **31.5** |
| p50 | 560 ms | 600 ms | 620 ms | 640 ms | **730 ms** |
| p95 | 640 ms | 670 ms | 700 ms | 770 ms | **830 ms** |

### Cause: GPU thermal throttling

`nvidia-smi` sampled every 2 s through run 2 (`results/gpu_telemetry_25u.csv`):

| | Idle | Under load |
|---|---:|---:|
| Temperature | 65 °C | **88 °C** |
| SM clock | 2520 MHz | **585–1000 MHz** |
| Power | 101 W | 231–339 W (350 W limit) |

`SwThermalSlowdown` was active in **45 of 60** load samples; the remaining 15
showed `SwPowerCap` near the 350 W limit. The SM clock drops roughly 60% below
its idle boost and stays there.

The L40S is a passively-cooled datacenter card that depends entirely on chassis
airflow, and this host does not supply enough of it. The correct reading:

- **~41 req/s** is what the software delivers on a cool L40S
- **~31 req/s** is what this chassis can sustain thermally

Nothing here indicates a software bottleneck — no error growth, no queue
collapse, no memory drift. The shorter FUNSD run (48.5 img/s) is less
heat-soaked and is the better measure of the engine itself.

### Methodology note

Locust reports ~36 req/s where the repo's async harness measured 40.6 img/s at
comparable concurrency. Part is thermal decay, part is Locust's own overhead —
it drives blocking `requests` under greenlets, so client-side cost is included
in these latencies. These are end-to-end client-observed numbers, which is
normally what a load test should report.

---

## Reproducing

Start the server (engines build on first run and then cache):

```bash
bash run_server.sh          # OCR_MODEL=medium, HTTP on :8080
curl -fsS http://localhost:8080/health/ready
```

Repo benchmarks:

```bash
cd tests/benchmark
python3 bench_throughput.py --server-url http://localhost:8080 --n 200
python3 bench_latency.py    --server-url http://localhost:8080 --n 60 --warmup 10
python3 bench_funsd_local.py   # needs FUNSD images staged, see below
```

FUNSD images are not in the repo. Stage the 50 test pages as
`compare-ocrs/funsd_cache/funsd_{000..049}.png`, ordered to match
`tests/benchmark/funsd_gt_words.json` — the order can be recovered by matching
each entry's word bag against the dataset's own annotation files, which yields a
unique assignment for all 50 pages.

Load test — concurrency, spawn rate and duration are stock Locust flags, so
nothing needs editing to scale:

```bash
pip install locust
locust -f locustfile.py --headless -u 25 -r 5 -t 2m --host http://localhost:8080
```

| Flag | Default | Purpose |
|---|---|---|
| `-u` / `--users` | — | concurrent users |
| `-r` / `--spawn-rate` | — | users started per second |
| `-t` / `--run-time` | — | duration (`90s`, `2m`, `1h`) |
| `--ocr-endpoint` | `/ocr/raw` | endpoint under test |
| `--ocr-image` | `business_letter.png` | single image to POST |
| `--ocr-image-dir` | — | directory of images, cycled round-robin |
| `--ocr-warmup` | `0` | per-user warmup requests, excluded from stats |

Larger example:

```bash
locust -f locustfile.py --headless -u 100 -r 10 -t 5m \
  --host http://localhost:8080 --ocr-warmup 2 \
  --ocr-image-dir tests/fixtures/images/png \
  --csv results_100u --html report_100u.html
```

Images are read into memory once at startup, so the test measures the server
rather than local disk.

---

## Files

| Path | Contents |
|---|---|
| `locustfile.py` | Configurable load test |
| `run_server.sh` | Server launch used for every run here |
| `results/throughput_medium.json` | Full endpoint × concurrency matrix |
| `results/bench_throughput.txt` | Matrix console output |
| `results/funsd.txt` | FUNSD accuracy + throughput |
| `results/results_25u_stats.csv` | Locust run 1 summary |
| `results/results_25u_stats_history.csv` | Run 1 per-second trend |
| `results/results_25u_run2_*.csv` | Run 2 (reproducibility check) |
| `results/gpu_telemetry_25u.csv` | `nvidia-smi` samples showing the throttle |
| `results/locust_report_25u.html` | Locust HTML report for run 1 (open in a browser) |

---

## Summary

1. **Published accuracy reproduces** — 92.34% FUNSD F1 against 91.9% claimed.
2. **Throughput scales with hardware as expected** — 48.5 img/s versus 86 on a
   5090, no evidence of a misconfigured build.
3. **The `medium` tier is not the headline number.** The 559 img/s banner figure
   is `tiny` on receipts on a 5090; `medium` on dense pages on an L40S is a
   different point on three separate axes. It is still far faster than the
   alternatives at higher accuracy.
4. **Sustained throughput here is thermally bound, not software bound** — worth
   checking cooling before concluding a deployment is slow.
5. **Plan for the cold start.** A ~3 hour first-boot engine build on non-Blackwell
   hardware needs a warm cache or a pre-baked volume in any real deployment;
   `TRT_OPT_LEVEL=3` reduces it substantially.
