# 40-user load test — native API vs PaddleX-compatible API (`medium` tier)

Companion to the [`small`-tier comparison](../l40s-ppocrv6-small/LOAD40.md), same
host and method. Both APIs driven to 40 concurrent users against the same
`OCR_MODEL=medium` backend, **run one at a time**, with the card cooled back to
57 °C between runs so neither inherited the other's heat.

**Result: on `medium` the PaddleX adapter costs nothing measurable.**

| Metric | Native `/ocr/raw` | PaddleX `/ocr` | Δ |
|---|---:|---:|---|
| Requests (2 min) | 4,281 | 4,302 | +0.5% |
| **Failures** | **0** | **0** | — |
| Throughput | **36.0 req/s** | **36.1 req/s** | +0.3% |
| **Mean latency** | **1077.2 ms** | **1072.4 ms** | **−4.8 ms** |
| p50 | 1000 ms | 1100 ms | 1 bucket |
| p95 / p99 | 1300 / 1300 ms | 1300 / 1300 ms | — |
| Min / Max | 132 / 1400 ms | 138 / 1407 ms | — |
| Response size | 15.9 KB | 20.5 KB | +29% |

HTML reports: [`native_40u.html`](results/load40/native_40u.html) ·
[`paddlex_40u.html`](results/load40/paddlex_40u.html) (open locally — GitHub
does not render HTML from the repo view).

### Read the mean, not the p50

Locust rounds percentiles to 100 ms buckets above one second, so the
1000 → 1100 ms p50 step is a **bucket boundary, not a measured regression**. The
unrounded mean moves the other way (1077.2 → 1072.4 ms). The honest reading is
no difference, within run-to-run noise.

At saturation, latency is queueing: 40 users ÷ 36 req/s ≈ 1.1 s per request. The
adapter's ~10 ms of JSON work is under 1% of that and disappears entirely.

---

## CPU

Attributed by process tree, sampled from `/proc` every 2 s. Host has **112
cores**; figures are cores.

| Process group | Native | PaddleX | Δ |
|---|---:|---:|---|
| `turboocr-server` | 2.37 cores | 2.32 cores | −0.05 |
| **PaddleX adapter** (4 uvicorn workers) | — | **0.58 cores** | **+0.58** |
| locust (load generator) | 0.09 cores | 0.11 cores | — |
| System total busy | 5.11 cores | 8.65 cores | +3.54 |

Resident memory: `turboocr-server` **5.14 GB**; adapter **319 MB** across its 4
workers.

**On the system-total row:** it rises by more than the adapter's 0.58 cores
because it also carries the resource sampler, shell polling, and kernel network
work for the larger payloads. Treat the per-process rows as the measurement and
the system total as an upper bound.

Two comparisons with `small` are worth noting. `turboocr-server` needs only
**2.37 cores** here against 3.88 on `small` — each request spends far longer on
the GPU and far less in host-side work. And the adapter costs **0.58 cores**
against 1.21, because it is handling a third of the request rate; per request
the work is unchanged.

---

## GPU

| Metric | Native | PaddleX | `small` reference |
|---|---:|---:|---:|
| Utilization (avg) | **99.7%** | **99.9%** | 93.1% |
| Memory | 39,283 MiB | 39,283 MiB | 27,152 MiB |
| Temperature (avg / peak) | 83.9 / 88 °C | 84.4 / 88 °C | 84.4 / 88 °C |
| SM clock (avg / min) | 1281 / 555 MHz | 1318 / 577 MHz | 2130 / 1065 MHz |
| Power (avg / peak) | 316 / 344 W | 318 / 343 W | 310 / 343 W |
| Samples thermally limited | 32 / 59 | 33 / 59 | 30 / 59 |

`medium` holds **39.3 GB** of VRAM against `small`'s 27.2 GB, and runs at
**1281 MHz average against 2130** — the heavier recognizer draws enough sustained
power that the card spends most of the run against its power and thermal limits,
bottoming at 555 MHz.

> **These are throttled numbers.** Both runs spent more than half their samples
> thermally or power limited, at roughly half the 2520 MHz boost clock. The
> comparison between the two APIs is fair — both started from 57 °C — but
> **36 req/s is this chassis's figure, not the L40S's.**

---

## Why `medium` hides the adapter completely

| Tier | GPU util | Native | PaddleX | Adapter cost |
|---|---:|---:|---:|---:|
| `small` | 93.1% | 114.6 req/s | 112.2 req/s | −2.1% |
| **`medium`** | **99.7%** | **36.0 req/s** | **36.1 req/s** | **0.0%** |

At 114 req/s on `small`, requests arrive fast enough that host-side cost
occasionally sits on the critical path. At 36 req/s on `medium`, the GPU *is* the
queue and everything else waits behind it.

**The heavier the model, the more free the compatibility layer becomes.** If you
are choosing between tiers, adapter overhead should not enter the decision on
`medium` at all, and is a rounding error on `small`.

---

## Reproducing

```bash
# 1 · native — no adapter running
python3 ../l40s-ppocrv6-small/sample_resources.py /tmp/res_native.json 140 &
locust -f locustfile.py --headless -u 40 -r 8 -t 2m \
  --host http://localhost:8080 --ocr-api native --ocr-warmup 2 \
  --html native_40u.html --csv native_40u

# 2 · cool back to the same starting temperature
until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 57 ]; do sleep 10; done

# 3 · PaddleX — start the adapter, then repeat
TURBO_OCR_URL=http://127.0.0.1:8080 uvicorn paddlex_adapter:app \
  --app-dir ../../compat/paddlex --host 127.0.0.1 --port 9000 --workers 4 &
python3 ../l40s-ppocrv6-small/sample_resources.py /tmp/res_paddlex.json 140 &
locust -f locustfile.py --headless -u 40 -r 8 -t 2m \
  --host http://127.0.0.1:9000 --ocr-api paddlex --ocr-warmup 2 \
  --html paddlex_40u.html --csv paddlex_40u
```

Switching the backend from `small` to `medium` between benchmark sets brought it
up **ready in 24 seconds with zero engine builds**, straight from the cached
TensorRT engines — an independent confirmation of the warm-cache path in
[DEPLOY.md](DEPLOY.md).

### One incident worth recording

A first PaddleX attempt was launched, appeared lost to a timeout, and was
relaunched — while the original was still running. Two load generators split the
GPU and the result read **17.7 req/s: an apparent 51% adapter penalty.** It was
discarded and re-measured cleanly.

The tell was arithmetic: 17.7 is almost exactly half of native's 36.0, and a
compatibility shim has no mechanism that would halve throughput while leaving the
error count at zero. This is why the instruction to run the two tests
sequentially is in every version of these notes — concurrent runs produce numbers
that look plausible and are worthless.

## Files

| Path | Contents |
|---|---|
| `results/load40/native_40u.html` | Locust HTML report, native API |
| `results/load40/paddlex_40u.html` | Locust HTML report, PaddleX API |
| `results/load40/*_stats.csv` | Summary per run |
| `results/load40/*_stats_history.csv` | Per-second trend |
| `results/load40/res_native.json` | CPU/GPU samples, native run |
| `results/load40/res_paddlex.json` | CPU/GPU samples, PaddleX run |
| `../l40s-ppocrv6-small/sample_resources.py` | The sampler |
