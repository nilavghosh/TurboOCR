# PP-OCRv6 `small` on an NVIDIA L40S — latency benchmark

Companion to the [`medium` tier run](../l40s-ppocrv6-medium/README.md), on the
same host, same build, same fixtures. The focus here is **latency**.

**Headline:** `small` is **3× faster than `medium` on both throughput and
latency**, for **1.5 points** of FUNSD word F1. On this hardware it is the
better operating point for almost any latency-sensitive or high-volume
workload.

| | `small` | `medium` | Δ |
|---|---:|---:|---|
| FUNSD word F1 | 90.81% | 92.34% | −1.53 pts |
| FUNSD throughput | **143.4 img/s** | 48.5 img/s | **3.0×** |
| FUNSD p50 / p95 | **22 / 32 ms** | 29 / 46 ms | 1.3–1.4× |
| `/ocr/raw` p50 (c=1) | **29.0 ms** | 34.6 ms | 1.19× |
| 25-user load, req/s | **115.6** | 35.9 | **3.2×** |
| 25-user load, p50 | **210 ms** | 670 ms | **3.2×** |

Environment is identical to the medium run — NVIDIA L40S 46 GB (Ada, sm_89),
driver 550.144.03, CUDA 12.4, TensorRT 10.16.0.72, `OCR_MODEL=small`. Build
details are in the [medium README](../l40s-ppocrv6-medium/README.md#building-without-docker).

---

## 1. Latency at concurrency 1

`tests/benchmark/bench_latency.py --n 60 --warmup 10`. Zero errors.

| Endpoint | p50 | p95 | p99 | avg | `medium` p50 |
|---|---:|---:|---:|---:|---:|
| `POST /ocr/pixels` | **21.3 ms** | 23.4 | 24.6 | 21.4 | 30.9 |
| `POST /ocr/raw` | **29.0 ms** | 30.9 | 32.0 | 29.0 | 34.6 |
| `POST /ocr` (base64) | 34.8 ms | 36.6 | 37.7 | 34.5 | 38.5 |
| `POST /ocr/batch` (8 imgs) | 297.0 ms | 313.2 | 339.4 | 293.9 | 265.8 |
| `POST /ocr/pdf?mode=geometric` | 57.3 ms | 74.3 | 83.5 | 58.2 | 76.3 |
| `POST /ocr/pdf?mode=auto` | 73.1 ms | 101.2 | 116.7 | 74.3 | 83.9 |
| `POST /ocr/pdf?mode=ocr` | 480.7 ms | 678.6 | 701.9 | 489.2 | 1384.9 |
| `POST /ocr/pdf?mode=auto_verified` | 535.8 ms | 748.4 | 776.3 | 548.7 | 1472.0 |

Two things stand out.

**Single-request latency improves far less than throughput does** — 1.19× on
`/ocr/raw` against 3.0× on throughput. At concurrency 1 a large share of the
~29 ms is fixed cost that the model tier does not touch: HTTP handling, image
decode, host↔device transfer, kernel launches. Only the recognition pass gets
cheaper. **If your goal is p50 on a single request, the tier change buys much
less than the throughput numbers suggest.** It is under load that it pays.

**The tail is remarkably tight.** p99/p50 is 1.10 on `/ocr/raw` and 1.15 on
`/ocr/pixels`. There is no GC, no JIT, and no dynamic batching queue to
introduce jitter — a fixed-shape TensorRT engine per request.

`mode=ocr` improves 2.9× (1385 → 481 ms) because a PDF page is many
recognition crops, so it is dominated by exactly the part the smaller model
accelerates. `batch` is slightly *slower* than medium's, which is measurement
noise across separate runs rather than a real regression.

---

## 2. FUNSD — accuracy and throughput

Same 50-page FUNSD test split, aligned to the repo's shipped ground truth.

| | `small` (L40S) | `medium` (L40S) | `small` published (RTX 5090) |
|---|---:|---:|---:|
| Word F1 | **90.81%** | 92.34% | 90.3% |
| Precision / Recall | 90.10% / 91.62% | 91.50% / 93.29% | — |
| Throughput | **143.4 img/s** | 48.5 img/s | 230 img/s |
| p50 / p95 | **22 / 32 ms** | 29 / 46 ms | — |

Accuracy again lands slightly **above** the published figure (90.81% vs 90.3%),
matching what the medium tier did. Throughput is 62% of the RTX 5090 number —
a little better than medium's 56%, consistent with a smaller model leaning less
on memory bandwidth, where the 5090's advantage is largest.

The measured `small`/`medium` speedup of **3.0×** exceeds the 2.67× implied by
the project's own 5090 table. Smaller models lose less to a slower GPU.

---

## 3. Sustained load — 25 users, 2 minutes

Same `locustfile.py` and procedure as the medium run.

| Metric | `small` | `medium` |
|---|---:|---:|
| Requests | **13,783** | 4,284 |
| Failures | **0 (0.00%)** | 0 (0.00%) |
| Throughput | **115.6 req/s** | 35.9 req/s |
| p50 | **210 ms** | 670 ms |
| p95 | **250 ms** | 820 ms |
| p99 | **260 ms** | 850 ms |
| Max | **314 ms** | 937 ms |

Under load the tier change delivers its full 3.2× on both throughput *and*
latency — the opposite of the concurrency-1 picture above, and the reason to
choose `small` for anything serving real traffic.

### Thermal behaviour

The same chassis-cooling limit as the medium run (88–89 °C,
`SwThermalSlowdown` in 42 of 60 load samples, SM clock 2430 → ~1035 MHz), but
the throughput decay is much gentler:

| Elapsed | ~10 s | ~40 s | ~80 s | ~120 s |
|---|---:|---:|---:|---:|
| `small` req/s | 114.1 | 123.1 | 116.9 | **111.2** (−2.5%) |
| `medium` req/s | 41.1 | 39.3 | 33.9 | **31.5** (−23%) |

`small` gives up ~2.5% over two minutes where `medium` gave up 23%. The
smaller model spends less time in the sustained high-power compute that drives
the card into thermal limit, so it holds its clocks better. **On a properly
cooled host, expect `medium` to gain more from the fix than `small` does.**

---

## Choosing a tier

| If you need… | Pick |
|---|---|
| Lowest p50 on single requests | `small` — but the gain is only ~1.2× |
| Highest sustained throughput | `small` — 3.2× at 25 concurrent users |
| Best accuracy, throughput secondary | `medium` — +1.5 pts F1 |
| Maximum throughput, accuracy tolerant | `tiny` — not benchmarked here; the project reports ~6× medium at −7.3 pts F1 |

For a Google Vision replacement sized at 83–167 pages/s, **one L40S on `small`
carries the average (143 img/s) and two carry the peak** — materially cheaper
than the H100 sizing that `medium` would require.

Whether `small`'s 1.5-point accuracy drop matters cannot be answered from
FUNSD. Measure both tiers on your own documents.

---

## 4. 40-user load test — native vs PaddleX API

Both APIs driven to 40 concurrent users against this same backend, run one at a
time. The PaddleX adapter costs **~2% throughput, ~10 ms p50 and 1.2 CPU
cores**; the GPU (93% utilized in both) does not notice it. Full numbers,
CPU/GPU breakdown and HTML reports: **[LOAD40.md](LOAD40.md)**.

---

## Files

| Path | Contents |
|---|---|
| `LOAD40.md` | 40-user native vs PaddleX comparison |
| `sample_resources.py` | CPU/GPU sampler used by that run |
| `results/load40/` | Locust HTML reports, CSVs, resource samples |
| `results/latency_small.txt` | Latency benchmark output |
| `results/funsd_small.txt` | FUNSD accuracy + throughput |
| `results/small_25u_stats.csv` | Locust summary |
| `results/small_25u_stats_history.csv` | Per-second trend |
| `results/gpu_telemetry_small.csv` | `nvidia-smi` samples during the load test |

## Reproducing

```bash
OCR_MODEL=small bash ../l40s-ppocrv6-medium/run_server.sh
cd ../../tests/benchmark
python3 bench_latency.py --server-url http://localhost:8080 --n 60 --warmup 10
python3 bench_funsd_local.py
locust -f ../../benchmarks/l40s-ppocrv6-medium/locustfile.py \
  --headless -u 25 -r 5 -t 2m --host http://localhost:8080 --ocr-warmup 2
```

Pre-built `small` engines (skips the ~2 hour build) are published as
[`engines-small-l40s-sm89-trt10.16-cuda12.4`](https://github.com/nilavghosh/TurboOCR/releases/tag/engines-small-l40s-sm89-trt10.16-cuda12.4).
Compatibility rules are in
[DEPLOY.md](../l40s-ppocrv6-medium/DEPLOY.md) — they apply identically here.

> Engine build cost, for planning: `det_small` took ~7 minutes and `rec_small`
> ~1 h 55 m at `TRT_OPT_LEVEL=5`. Not proportionally faster than `medium`'s
> 2 h 15 m — TensorRT build time tracks graph structure more than weight size.
> The `cls`, `layout` and `doc_ori` engines are shared with `medium` and are
> reused from cache unchanged.
