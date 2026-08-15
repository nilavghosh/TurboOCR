# PaddleX API compatibility — differences

`paddlex_adapter.py` puts a PaddleX-shaped façade in front of TurboOCR so an
existing PaddleX client can be repointed by changing a hostname. The wire
format matches; the engine underneath does not. This file records every place
the two diverge, so nothing surprises you in production.

Schema was taken from the **PaddleX 3.7.2** source, not from documentation:

| Source file | What it fixes |
|---|---|
| `paddlex/inference/serving/schemas/ocr.py` | request fields, `OCRResult`, `InferResult` |
| `paddlex/inference/serving/schemas/shared/ocr.py` | `file`, `fileType` |
| `paddlex/inference/serving/infra/models.py` | response envelope, `dataInfo` |
| `paddlex/inference/pipelines/ocr/result.py` | `prunedResult` keys and their order |
| `.../_pipeline_apps/_common/common.py` | `prune_result()` drops `input_path`, `page_index` |

> **Verify against your own version.** The adapter was written against 3.7.2
> and tested against TurboOCR, but **not** diffed against a live PaddleX server
> — none was available on the benchmark host. Before cutting over, run the
> checklist in [Validating the swap](#validating-the-swap) against your actual
> deployment.

---

## The difference that matters most

**The models are different.** TurboOCR runs PP-OCRv6; your PaddleX deployment
almost certainly runs PP-OCRv4 or v5. Recognized strings, confidence values,
and box coordinates **will not be byte-identical**, and no amount of API
shimming changes that. Everything below is about the envelope; this is about
the contents.

If anything downstream depends on exact text, exact scores, or exact geometry —
golden-file tests, checksums, cached keyed-by-output — it needs revalidating
against real documents. See the accuracy comparison in
[`benchmarks/l40s-ppocrv6-medium/README.md`](../../benchmarks/l40s-ppocrv6-medium/README.md):
92.34% FUNSD word F1, competitive but not identical.

---

## Endpoints

| PaddleX endpoint | Status |
|---|---|
| `POST /ocr` | **Implemented** |
| `GET /health` | **Implemented** — also probes the backend, so it reports 503 when TurboOCR is not ready rather than a bare liveness OK |
| `POST /layout-parsing` (PP-StructureV3) | Not implemented |
| `POST /chatocr-*`, `/doc-understanding`, other pipelines | Not implemented |

TurboOCR's native routes (`/ocr/raw`, `/ocr/pdf`, `/ocr/batch`, `/metrics`, …)
remain available on the backend port (8081 by default) and are unaffected.

---

## Request fields

| Field | Status | Behaviour |
|---|---|---|
| `file` | **Supported** | base64 or `http(s)://` URL, same as PaddleX |
| `fileType` | **Supported** | `0` = PDF, `1` = image; `null` infers from URL extension, else 422 |
| `logId` | **Supported** | echoed back; generated when absent |
| `textRecScoreThresh` | **Supported** | applied by the adapter — lines below the threshold are dropped from `rec_*` but kept in `dt_polys`, matching PaddleX semantics |
| `useTextlineOrientation` | **Server-level** | reflected in `model_settings`, but the classifier is configured at startup (`DISABLE_ANGLE_CLS`, `CLS_ALL_BOXES`), not per request |
| `useDocOrientationClassify` | **Server-level** | TurboOCR exposes page orientation only on `/ocr/pdf?autorotate=1`, not per request |
| `textDetLimitSideLen` | **Ignored** | baked into the TensorRT engine profile at build time |
| `textDetLimitType` | **Ignored** | server-level (`DET_LIMIT_TYPE`) |
| `textDetThresh` | **Ignored** | server-level (`DET_DB_THRESH`) |
| `textDetBoxThresh` | **Ignored** | server-level (`DET_BOX_THRESH`) |
| `textDetUnclipRatio` | **Ignored** | server-level (`DET_UNCLIP`) |
| `useDocUnwarping` | **Unsupported** | no unwarping model exists in TurboOCR |
| `returnWordBox` | **Unsupported** | recognition is line-level; there are no per-word boxes to return |
| `visualize` | **Unsupported** | annotated images are never rendered (see [Image fields](#image-fields)) |

### Why the detection knobs cannot be per-request

TurboOCR compiles detection into a TensorRT engine whose optimization profile
encodes the max side length, and the cache key includes `DET_MAX_SIDE` and
`DET_OPT_BATCH`. Honouring `textDetLimitSideLen` per request would mean a
distinct engine per value — hours of build time each. The thresholds
(`thresh`, `box_thresh`, `unclip_ratio`) are read once at startup into the
detector's config. Set them as environment variables on the container instead;
the adapter echoes the effective values back in `text_det_params` so the
response always reports what actually ran, never what was asked for.

### Strict mode

By default, unsupported and server-level parameters are **silently ignored** —
maximum compatibility, at the cost of a caller believing a knob took effect
when it did not. Set `PADDLEX_STRICT_PARAMS=1` to reject any request carrying
them with a 422 naming the offending fields. Recommended while migrating: it
turns a silent behavioural difference into a loud one, and you can turn it off
once the callers are clean.

---

## Response

The envelope is identical:

```json
{"logId": "...", "errorCode": 0, "errorMsg": "Success",
 "result": {"ocrResults": [...], "dataInfo": {...}}}
```

### `prunedResult`

Keys are emitted in PaddleX's own order. `input_path` and `page_index` are
correctly **absent** — `prune_result()` strips them, so a compatible payload
must not carry them.

| Key | Status |
|---|---|
| `model_settings` | Present. `use_doc_preprocessor` is always `false`; `use_textline_orientation` reflects the request |
| `dt_polys` | Present — every detected quad |
| `text_det_params` | Present — **effective** server values, not the requested ones |
| `text_type` | Always `"general"`. Seal recognition (`"seal"`) is not supported |
| `textline_orientation_angles` | **Omitted by default** — see below |
| `text_rec_score_thresh` | Present — the threshold actually applied |
| `return_word_box` | Always `false` |
| `rec_texts` / `rec_scores` / `rec_polys` | Present — post-threshold |
| `rec_boxes` | Present — **derived**, see below |
| `doc_preprocessor_res` | **Never present**, consistent with `use_doc_preprocessor: false` |

**`textline_orientation_angles` is omitted rather than faked.** TurboOCR runs a
0°/180° classifier but does not report the per-line decision in its response.
The key is optional in PaddleX's own output (`if "textline_orientation_angles"
in self`), so omitting it is schema-valid and honest; emitting zeros would be a
fabricated per-line claim. If a downstream consumer requires the key to exist,
set `PADDLEX_EMIT_ORIENTATION_ANGLES=1` to emit `0` for every line — but treat
those zeros as padding, not data.

**`rec_boxes` is derived, not native.** PaddleX gets axis-aligned boxes from
the detector; the adapter computes the axis-aligned hull of each quad
(`min/max` over the four corners). For horizontal text these agree closely. For
rotated or skewed lines the hull is **larger** than a true fitted box. If you
depend on `rec_boxes` for tight cropping on rotated text, use `rec_polys`
instead.

**`dt_polys` and `rec_polys` are the same set** unless `textRecScoreThresh`
filters some out. TurboOCR does not emit detections that failed recognition, so
the "detected but not recognized" population that PaddleX can show is not
observable here.

### Image fields

`ocrImage`, `docPreprocessingImage` and `inputImage` are **always `null`**.
Rendering annotated output would mean decoding and re-encoding every page on
the CPU — precisely the cost this backend exists to avoid. `visualize: true` is
accepted and ignored (or rejected under strict mode). PaddleX's file-storage
and `return_urls` behaviour has no counterpart.

### `dataInfo`

| Input | Emitted |
|---|---|
| Image | `{"width": W, "height": H, "type": "image"}` |
| PDF | `{"numPages": N, "pages": [{"width","height"}...], "type": "pdf"}` |
| TIFF | Treated as a **single image**, not as `type: "tiff"` |

PaddleX expands multi-page TIFF into one entry per page and reports
`TIFFInfo`. The adapter treats TIFF as an ordinary image and processes only the
first page. Convert multi-page TIFF to PDF upstream if you rely on this.

PDF page dimensions come from the render at `PADDLEX_PDF_DPI` (default 100),
so they are **pixel** dimensions at that DPI, not PDF points. PaddleX's own
values depend on its render DPI too, so expect these to differ numerically
unless the DPI matches.

---

## Errors

| Condition | HTTP | `errorCode` |
|---|---|---|
| Malformed JSON, schema violation, bad base64, undeterminable file type, page-count over `MAX_NUM_INPUT_IMGS` | 422 | 422 |
| Unsupported parameters, strict mode only | 422 | 422 |
| Backend error or unreachable | 500 | 500 |
| Backend not ready (`/health`) | 503 | 503 |

`errorCode` mirrors the HTTP status, as PaddleX's own handlers do. **Error
message strings differ** — do not match on `errorMsg` text.

---

## Operational differences

**PDF handling.** PaddleX rasterizes every page and OCRs it. The adapter uses
`mode=auto`, which extracts the embedded text layer when one is present and
only rasterizes when it is not. That is far faster — ~280 pages/s versus ~11 —
and for born-digital PDFs the text comes from the PDF itself rather than from
OCR, so it is *more* accurate, not less. It does mean output for such pages
reflects the document's own text layer. Set `PADDLEX_PDF_DPI` to control render
DPI; force full OCR by editing `_run_pdf` to use `mode=ocr` if you need
byte-comparable behaviour with PaddleX.

**Concurrency.** The adapter is stateless; scale it with `PADDLEX_WORKERS`
(default 4). The GPU backend saturates around concurrency 4 and holds flat to
32, so the adapter is not the bottleneck.

**Model tier** is chosen at container start with `OCR_MODEL`
(`tiny`/`small`/`medium`), not per request. PaddleX model selection via
pipeline config has no request-level equivalent.

---

## Validating the swap

Run this against both your PaddleX deployment and this adapter, then diff:

```bash
B64=$(base64 -w0 sample.png)
for HOST in http://paddlex:8080 http://turboocr:8080; do
  curl -s -X POST "$HOST/ocr" -H 'Content-Type: application/json' \
    -d "{\"file\": \"$B64\", \"fileType\": 1}" \
  | python3 -m json.tool > "$(basename $HOST).json"
done
diff paddlex:8080.json turboocr:8080.json
```

Expect differences in `rec_texts`, `rec_scores` and coordinates — those are the
models disagreeing, which is the real thing to evaluate. What should **not**
differ is the structure: key names, nesting, types. Check specifically that

1. your client tolerates `ocrImage: null`,
2. your client tolerates a missing `textline_orientation_angles`,
3. nothing reads `doc_preprocessor_res`,
4. nothing depends on per-request detection thresholds taking effect,
5. `rec_boxes` is not used for tight crops on rotated text.

Then measure accuracy on **your** documents, not on FUNSD. A public benchmark
says nothing about your invoices.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TURBO_OCR_URL` | `http://127.0.0.1:8081` | backend address |
| `PADDLEX_PORT` | `8080` | adapter port (PaddleX's default) |
| `TURBO_PORT` | `8081` | private backend port |
| `PADDLEX_WORKERS` | `4` | uvicorn workers |
| `PADDLEX_STRICT_PARAMS` | `0` | 422 on unsupported params instead of ignoring |
| `PADDLEX_EMIT_ORIENTATION_ANGLES` | `0` | emit placeholder zeros for the optional key |
| `PADDLEX_PDF_DPI` | `100` | PDF render DPI |
| `PADDLEX_TIMEOUT_S` | `120` | backend request timeout |
| `MAX_NUM_INPUT_IMGS` | `1000` | page-count ceiling per request |
| `DET_*` | see table | echoed into `text_det_params`; set to match the server's real config |

## Running

```bash
docker build -f compat/paddlex/Dockerfile.paddlex \
  --build-arg CUDA_ARCH=89 -t turboocr:paddlex-sm89 .

# Engines are resolved out of /models by GPU + toolchain automatically.
docker run --gpus all -p 8080:8080 \
  -v turboocr-models:/models -e OCR_MODEL=medium \
  turboocr:paddlex-sm89
```

See [`deploy/README.md`](../../deploy/README.md) for the volume layout and the
full environment/build-arg matrix.

Standalone, against an existing TurboOCR server:

```bash
pip install -r compat/paddlex/requirements.txt
TURBO_OCR_URL=http://localhost:8080 \
  uvicorn paddlex_adapter:app --app-dir compat/paddlex --host 0.0.0.0 --port 9000
```
