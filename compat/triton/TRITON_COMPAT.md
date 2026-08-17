# Triton / KServe v2 API compatibility — differences

`compat/triton/triton_adapter.py` puts the KServe v2 inference protocol in
front of TurboOCR so an existing Triton client can be repointed with a
hostname change. It speaks the HTTP protocol including the **binary tensor
data extension**, which is what `tritonclient.http` uses by default.

Protocol references this was written against:

| Source | Used for |
|---|---|
| KServe v2 predict protocol | endpoint set, request/response envelopes |
| Triton `extension_binary_data.md` | `Inference-Header-Content-Length`, BYTES framing |
| Triton `extension_model_repository.md` | `POST /v2/repository/index` |

> **Verified against `tritonclient.http`** (metadata, config, readiness,
> binary and JSON inference, batching) and by the protocol suite in
> `test_adapter_offline.py`. It has **not** been diffed against a live Triton
> server serving the same model.

---

## The difference that matters most

**There is no gRPC.** Triton's gRPC service (`inference.GRPCInferenceService`,
port 8001) is not implemented — only HTTP. A client constructed with
`tritonclient.grpc.InferenceServerClient` will not connect. Switching a Python
client is usually a two-line change:

```python
# before
import tritonclient.grpc as tc; cl = tc.InferenceServerClient("host:8001")
# after
import tritonclient.http as tc; cl = tc.InferenceServerClient("host:8000")
```

TurboOCR does have its own gRPC service on 50051, but it speaks
`proto/ocr.proto`, not Triton's protocol. It is a different API, not a
drop-in.

**The tensor names are not your old model's.** This adapter defines its own
signature (`IMAGE` → `OCR_RESULT`, `TEXT`). If your client hard-codes the
names your previous Triton model used, either change the client or rename the
tensors here with `TRITON_INPUT_NAME` / `TRITON_OUTPUT_NAME` /
`TRITON_TEXT_OUTPUT_NAME`. The model name and version are configurable for the
same reason and default to `ocr` / `1`.

---

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /v2` | ✅ server metadata |
| `GET /v2/health/live` | ✅ always 200 while the process is up |
| `GET /v2/health/ready` | ✅ proxies the backend's `/health/ready` |
| `GET /v2/models/{m}/ready`, `…/versions/{v}/ready` | ✅ |
| `GET /v2/models/{m}`, `…/versions/{v}` | ✅ model metadata |
| `GET /v2/models/{m}/config` | ✅ minimal config |
| `POST /v2/models/{m}/infer`, `…/versions/{v}/infer` | ✅ |
| `POST /v2/repository/index` | ✅ single static entry |
| `POST /v2/repository/models/{m}/load` / `unload` | ❌ the model set is fixed at container start |
| `GET /v2/models/{m}/stats` | ❌ use TurboOCR's own `/metrics` |
| `GET /v2/logging`, `/v2/trace/setting` | ❌ |
| System / CUDA shared memory | ❌ |
| gRPC | ❌ HTTP only |

`GET /v2` advertises only `binary_tensor_data`, `model_repository` and
`model_configuration` — the extensions that are actually implemented. A client
that probes for shared memory is told no here rather than failing later.

---

## Tensors

```
input   IMAGE       BYTES  [-1]   one encoded image or PDF per element
output  OCR_RESULT  BYTES  [-1]   one JSON document per input element
output  TEXT        BYTES  [-1]   recognized text per input element
```

`OCR_RESULT` elements are TurboOCR's native per-image response verbatim —
`{"results": [...], "layout": [...], "tables": [...], ...}`, exactly the shape
`docs/api/http.md` documents. A PDF element yields `{"pages": [...]}` instead.

`TEXT` elements are the recognized text with lines newline-joined; PDF pages
are separated by a form feed (`\f`), the convention `pdftotext` established.

A request that omits `outputs` gets both, which is Triton's behaviour.

### BYTES encoding, and the one asymmetry

**Binary (what `tritonclient` sends).** Standard framing: per element, a 4-byte
little-endian length followed by the raw bytes, with
`Inference-Header-Content-Length` separating the JSON header from the tensor
blob. Fully supported both directions.

**JSON.** Triton represents a BYTES tensor as an array of strings. That works
for output — both of ours are UTF-8 by construction — so JSON responses carry
plain strings, exactly like Triton.

It cannot work for **input**, because a PNG has no string form. JSON input
elements must therefore be **base64**:

```bash
curl -X POST http://localhost:8000/v2/models/ocr/infer \
  -H 'Content-Type: application/json' -d '{
    "inputs": [{"name":"IMAGE","datatype":"BYTES","shape":[1],
                "data":["'"$(base64 -w0 page.png)"'"]}],
    "outputs": [{"name":"TEXT"}]}'
```

This is a deliberate deviation and the only one in the data path. Binary-mode
clients never encounter it.

### When is the response binary?

1. An explicit per-output `parameters.binary_data` wins.
2. Then the request-level `parameters.binary_data_output`.
3. Otherwise it **mirrors the request**: a request that used
   `Inference-Header-Content-Length` gets binary back, a plain-JSON request
   gets JSON.

Rule 3 is a choice, not the spec. It means `tritonclient` (which always sets
the flag explicitly anyway) is unaffected, while `curl` with a JSON body gets
something readable instead of a binary blob.

---

## Parameters

Per-request options travel in the KServe `parameters` object, so the model
signature stays fixed rather than growing a tensor per flag:

| Parameter | Type | Maps to |
|---|---|---|
| `layout` | bool | `?layout=1` |
| `reading_order` | bool | `?reading_order=1` |
| `as_blocks` | bool | `?as_blocks=1` |
| `tables` | bool | `?tables=1` |
| `formulas` | bool | `?formulas=1` |
| `text` | bool | `?text=0` to skip OCR |
| `pdf_mode` | string | `/ocr/pdf?mode=` (default `auto`) |
| `pdf_dpi` | int | `/ocr/pdf?dpi=` (default 100) |

```python
result = client.infer("ocr", [inp], parameters={"layout": True, "tables": True})
```

The same startup gates apply as on the native API: `tables`/`formulas` need
their backend configured, and any layout-derived flag against a
`DISABLE_LAYOUT=1` server returns the backend's `400 LAYOUT_DISABLED` — passed
through with its status intact rather than laundered into a 500.

**Protocol reserved words** (`sequence_id`, `sequence_start`, `sequence_end`,
`priority`, `timeout`, `binary_data_output`, `classification`, the shared-memory
keys) are accepted and ignored. Rejecting `sequence_id` on a stateless model
would break clients that set it unconditionally.

**Anything else** is ignored with a log line, or rejected with 400 when
`TRITON_STRICT_PARAMS=1` — the same convention the PaddleX adapter uses.

---

## Batching

An input tensor with N elements is one request carrying N images.

- **N = 1** → the backend's `/ocr/raw`.
- **N > 1, all images** → the backend's `/ocr/batch`, which uses nvJPEG batch
  decode when ≥2 inputs are JPEG.
- **Any PDF in the batch** → sequential single calls, because `/ocr/batch` has
  no PDF slot. Correct, just slower; keep PDFs in their own request.

`TRITON_MAX_BATCH_SIZE` (default 64) bounds N; a larger tensor is rejected with
400 rather than accepted and queued.

**Per-slot failures stay in-band.** A Triton response tensor must have one
element per input, so a slot the backend could not decode comes back as
`{"error": "decode_failed"}` in that position instead of being dropped. The
batch never desynchronises.

Note that `max_batch_size` in `GET /v2/models/ocr/config` is reported as `0`.
That is the honest answer: batching happens inside TurboOCR's pipeline pool,
not in a Triton scheduler arranging requests, so there is no dynamic batcher to
describe.

---

## Errors

Triton's envelope, `{"error": "..."}` with the status on the response:

| Status | When |
|---|---|
| 400 | malformed request, wrong datatype, shape mismatch, truncated binary tensor, unknown output, batch over the cap, strict-mode parameter rejection |
| 400 | backend 4xx passed through (`LAYOUT_DISABLED`, `TABLE_BACKEND_DISABLED`, …) with its message |
| 404 | unknown model or version |
| 500 | backend 5xx or an unhandled adapter failure |
| 503 | backend unreachable |

`/v2/health/ready` returns 400 when the backend is not ready, matching Triton.

---

## Operational differences

**Readiness reflects the backend, not the adapter.** `/v2/health/live` is up as
soon as the process is; `/v2/health/ready` only passes once
turboocr-server answers its own readiness probe. On a cold TensorRT cache that
gap is hours — point your k8s readiness probe at `/v2/health/ready` and your
liveness probe at `/v2/health/live`, or the pod gets killed mid-engine-build.

**No model management.** The model set is fixed at container start, and the
model *tier* is chosen with `OCR_MODEL` (`tiny`/`small`/`medium`), not per
request. Triton's load/unload endpoints have no equivalent — run a second
container to serve a second tier.

**Concurrency.** The adapter is stateless; scale it with `TRITON_WORKERS`
(default 4). The GPU backend saturates around concurrency 4 and holds flat to
32, so the adapter is not the bottleneck.

**Metrics.** Triton's Prometheus endpoint on 8002 is not reproduced. TurboOCR
has its own `/metrics` on the backend port — see `docs/api/monitoring.md`.

---

## Validating the swap

Protocol suite, no GPU or model needed (the backend is a mock transport):

```bash
python3 compat/triton/test_adapter_offline.py
```

End-to-end against a live server:

```bash
ADAPTER_URL=http://localhost:8000 python3 compat/triton/test_adapter_live.py
```

Smoke test with the real client:

```python
import numpy as np, tritonclient.http as httpclient
cl = httpclient.InferenceServerClient("localhost:8000")
assert cl.is_model_ready("ocr")

data = np.array([open("page.png","rb").read()], dtype=object)
inp = httpclient.InferInput("IMAGE", [1], "BYTES")
inp.set_data_from_numpy(data, binary_data=True)

r = cl.infer("ocr", [inp], outputs=[httpclient.InferRequestedOutput("TEXT")])
print(r.as_numpy("TEXT")[0].decode())
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TURBO_OCR_URL` | `http://127.0.0.1:8081` | backend base URL |
| `TRITON_PORT` | `8000` | adapter listen port (Triton's default) |
| `TRITON_WORKERS` | `4` | uvicorn workers |
| `TRITON_MODEL_NAME` | `ocr` | model name in the URL path |
| `TRITON_MODEL_VERSION` | `1` | the single served version |
| `TRITON_SERVER_VERSION` | `2.44.0` | reported by `GET /v2` |
| `TRITON_INPUT_NAME` | `IMAGE` | input tensor name |
| `TRITON_OUTPUT_NAME` | `OCR_RESULT` | JSON output tensor name |
| `TRITON_TEXT_OUTPUT_NAME` | `TEXT` | text output tensor name |
| `TRITON_MAX_BATCH_SIZE` | `64` | elements per input tensor |
| `TRITON_STRICT_PARAMS` | `0` | `1` rejects unknown parameters with 400 |
| `TRITON_TIMEOUT_S` | `120` | backend request timeout |
| `TRITON_PDF_MODE` / `TRITON_PDF_DPI` | `auto` / `100` | PDF defaults |
| `TRITON_LOG_LEVEL` | `info` | uvicorn log level |

`PADDLEX_API=1` and `TRITON_API=1` are mutually exclusive — the entrypoint
fails fast rather than letting the second adapter lose a race for the port.

---

## Running

```bash
# CUDA 12.8 base: sm_120 (default) and sm_90 from one recipe.
docker build -f compat/triton/Dockerfile.triton \
  --build-arg CUDA_ARCH=120 -t turboocr:triton-sm120 .   # RTX PRO 6000
docker build -f compat/triton/Dockerfile.triton \
  --build-arg CUDA_ARCH=90  -t turboocr:triton-sm90 .    # H100

# Engines are resolved out of /models by GPU + toolchain automatically.
docker run --gpus all -p 8000:8000 \
  -v turboocr-models:/models -e OCR_MODEL=medium \
  turboocr:triton-sm120
```

See [`deploy/README.md`](../../deploy/README.md) for the volume layout, engine
warming, and the full environment/build-arg matrix.
