# 40-user load test — native API vs PaddleX-compatible API

Both APIs driven to 40 concurrent users against the **same** backend
(`OCR_MODEL=small`, NVIDIA L40S), **run one at a time** so neither competes for
the GPU. Same fixture, same duration, same load generator.

**Result: the PaddleX adapter costs ~2% throughput, ~10 ms of p50, and 1.2 CPU
cores. The GPU does not notice it.**

| Metric | Native `/ocr/raw` | PaddleX `/ocr` | Δ |
|---|---:|---:|---|
| Requests (2 min) | 13,665 | 13,376 | −2.1% |
| **Failures** | **0** | **0** | — |
| Throughput | **114.6 req/s** | **112.2 req/s** | −2.1% |
| p50 | 340 ms | 350 ms | +10 ms |
| p95 | 390 ms | 400 ms | +10 ms |
| p99 | 410 ms | 420 ms | +10 ms |
| Max | 475 ms | 468 ms | −7 ms |

HTML reports: [`native_40u.html`](results/load40/native_40u.html) ·
[`paddlex_40u.html`](results/load40/paddlex_40u.html)
(open locally — GitHub does not render HTML from the repo view).

---

## CPU

Sampled every 2 s from `/proc`, attributed **by process tree** so uvicorn's
`spawn_main` workers are credited to the adapter rather than lost. Host has
**112 cores**; figures are cores, not percent-of-one-core.

| Process group | Native | PaddleX | Δ |
|---|---:|---:|---|
| `turboocr-server` | 3.88 cores | 3.82 cores | −0.06 |
| **PaddleX adapter** (4 uvicorn workers) | — | **1.21 cores** | **+1.21** |
| locust (load generator) | 0.27 cores | 0.25 cores | — |
| **System total busy** | **5.26 cores** | **6.35 cores** | **+1.09** |

Resident memory: `turboocr-server` **6.74 GB** in both runs; the adapter adds
**298 MB** across its 4 workers.

The striking number is how little CPU any of this uses: **4.7% of the machine**
serving 114 req/s. TurboOCR holds steady at ~3.9 cores whether or not the
adapter is in front of it, which confirms the adapter is pure additive overhead
and does not perturb the backend.

**Where the adapter's 1.2 cores go:** JSON parsing and base64-decoding a
~280 KB envelope per request, then re-serializing a larger response. The
PaddleX wire format inflates the request ~33% (base64) and the response from
14.5 KB to 18.6 KB (`prunedResult` repeats each polygon in `dt_polys` and
`rec_polys`, and adds `rec_boxes`). That is inherent to the format, not to this
implementation.

**Scaling note:** at 112 req/s the adapter needs ~1.2 cores, i.e. roughly
**0.011 cores per request/s**. Budget ~1 core per 100 req/s, and raise
`PADDLEX_WORKERS` if you push a single container much beyond that.

---

## GPU

| Metric | Native | PaddleX |
|---|---:|---:|
| Utilization | 93.1% avg (98% peak) | 93.6% avg (99% peak) |
| Memory | 27,152 MiB | 27,152 MiB |
| Temperature | 84.4 °C avg, **88 °C peak** | 86.4 °C avg, **88 °C peak** |
| SM clock | 2130 MHz avg (min 1065) | 2033 MHz avg (min 1500) |
| Power | 310 W avg (343 W peak) | 299 W avg (335 W peak) |

**The GPU is the bottleneck in both runs** — 93% utilization against 4.7% CPU.
That is the correct shape for this workload and it explains why the adapter
costs so little end to end: it adds CPU work to a system that has CPU to spare,
while the GPU stays equally busy.

### Thermal throttling, again

Both runs hit the same chassis-cooling limit documented in the
[medium-tier run](../l40s-ppocrv6-medium/README.md#cause-gpu-thermal-throttling):
88 °C peak, SM clock dropping from a 2520 MHz boost to as low as 1065 MHz.

| Throttle reason (of 59 load samples) | Native | PaddleX |
|---|---:|---:|
| `SwPowerCap` | 29 | 14 |
| `SwThermalSlowdown` | 30 | 45 |

The PaddleX run spent more samples thermally limited and fewer power-limited,
and drew 11 W less on average — the card was hotter, so it clocked down sooner
and consequently drew less power. This is a **run-order artifact**: the PaddleX
test ran second, on a card already heat-soaked from the native test. It is not
a property of the adapter.

**This matters for reading the 2.1% throughput gap: part of it is thermal, not
adapter overhead.** The true adapter cost is *at most* 2.1% and plausibly less.
Both runs are also throttle-limited overall, so absolute throughput here is
below what a properly cooled L40S would deliver.

---

## Reproducing

```bash
# 1. native — stop the adapter first so nothing else competes
python3 benchmarks/l40s-ppocrv6-small/sample_resources.py /tmp/res_native.json 140 &
locust -f benchmarks/l40s-ppocrv6-medium/locustfile.py --headless \
  -u 40 -r 8 -t 2m --host http://localhost:8080 \
  --ocr-api native --ocr-warmup 2 --html native_40u.html --csv native_40u

# 2. PaddleX — start the adapter, then repeat
TURBO_OCR_URL=http://127.0.0.1:8080 uvicorn paddlex_adapter:app \
  --app-dir compat/paddlex --host 127.0.0.1 --port 9000 --workers 4 &
python3 benchmarks/l40s-ppocrv6-small/sample_resources.py /tmp/res_paddlex.json 140 &
locust -f benchmarks/l40s-ppocrv6-medium/locustfile.py --headless \
  -u 40 -r 8 -t 2m --host http://127.0.0.1:9000 \
  --ocr-api paddlex --ocr-warmup 2 --html paddlex_40u.html --csv paddlex_40u
```

`--ocr-api paddlex` makes the locustfile post the PaddleX JSON envelope
(base64 `file` + `fileType`) to `/ocr` and validate `errorCode` /
`ocrResults[].prunedResult` in the response — a PaddleX error returned with
HTTP 200 is counted as a failure, not a success.

Run them **sequentially**. Concurrent runs share one GPU and both sets of
numbers become meaningless.

## Files

| Path | Contents |
|---|---|
| `results/load40/native_40u.html` | Locust HTML report, native API |
| `results/load40/paddlex_40u.html` | Locust HTML report, PaddleX API |
| `results/load40/*_stats.csv` | Summary per run |
| `results/load40/*_stats_history.csv` | Per-second trend |
| `results/load40/res_native.json` | CPU/GPU samples, native run |
| `results/load40/res_paddlex.json` | CPU/GPU samples, PaddleX run |
| `sample_resources.py` | The sampler |

> The sampler attributes CPU by walking the process tree from a matched root.
> An earlier cmdline-matching version credited the adapter **0.00 cores**,
> because uvicorn workers are `multiprocessing.spawn_main` children whose
> cmdline mentions neither "uvicorn" nor the app module. Both runs here were
> re-measured with the fixed version.
