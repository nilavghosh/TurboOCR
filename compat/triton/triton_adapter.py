"""Triton-compatible (KServe v2) inference API in front of TurboOCR.

Speaks the KServe v2 / Triton HTTP inference protocol, including the binary
tensor data extension, and translates each `infer` call into TurboOCR's native
HTTP API. An existing Triton client — `tritonclient.http`, or anything that
POSTs to `/v2/models/<name>/infer` — can be repointed with a hostname change.

Protocol references:
  KServe v2 predict protocol   https://kserve.github.io/website/modelserving/data_plane/v2_protocol/
  Triton binary tensor data    docs/protocol/extension_binary_data.md
  Triton model metadata        docs/protocol/README.md

Tensor contract (names are configurable, see the env table in TRITON_COMPAT.md).

The DEFAULT is PaddleX-HPS compatibility mode (TRITON_PADDLE_ENVELOPE=1), a
drop-in for the paddle-ocr-triton `ocr` model so an unmodified client such as
inocr-client works by only changing the endpoint host:

    input   input   BYTES [-1]   one PaddleX envelope per element:
                                 {"file": <base64 image|pdf>, "fileType": 0|1,
                                  "visualize": bool}  (a raw encoded image is
                                 also accepted)
    output  output  BYTES [-1]   PaddleX-HPS JSON envelope per element:
                                 {errorCode, errorMsg, result:{ocrResults:
                                  [{prunedResult:{rec_texts, rec_scores,
                                    rec_polys, dt_polys, ...}}], dataInfo}}
    output  TEXT    BYTES [-1]   recognized text per element (convenience)

Set TRITON_PADDLE_ENVELOPE=0 for the native contract instead (input `IMAGE` =
one encoded image per element; output `OCR_RESULT` = TurboOCR's native doc JSON
plus `TEXT`). Tensor names default to `input`/`output` here and are overridable
via TRITON_INPUT_NAME / TRITON_OUTPUT_NAME regardless of mode.

Per-request options travel in the KServe `parameters` object rather than as
extra tensors, so the model signature stays fixed: `layout`, `reading_order`,
`as_blocks`, `tables`, `formulas`, `text`, `pdf_mode`, `pdf_dpi`. Unknown
parameters are ignored by default; TRITON_STRICT_PARAMS=1 rejects them instead
of silently doing something other than what the caller asked.

Not everything Triton exposes exists here — no gRPC, no shared memory, no
dynamic batching knobs, no model repository control. See TRITON_COMPAT.md for
the endpoint-by-endpoint differences.

Run:
    uvicorn triton_adapter:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger("triton_adapter")

TURBO_URL = os.environ.get("TURBO_OCR_URL", "http://127.0.0.1:8081").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("TRITON_TIMEOUT_S", "120"))
STRICT_PARAMS = os.environ.get("TRITON_STRICT_PARAMS", "0").lower() in ("1", "true", "yes", "on")

# Defaults chosen so a deployment replacing a Triton server whose model was
# called "ocr" version 1 needs no client change at all.
MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "ocr")
MODEL_VERSION = os.environ.get("TRITON_MODEL_VERSION", "1")
SERVER_VERSION = os.environ.get("TRITON_SERVER_VERSION", "2.44.0")

# PaddleX-HPS compatibility is the default: tensor names and payload shapes
# mirror the paddle-ocr-triton `ocr` model so inocr-client works unchanged.
# Set TRITON_PADDLE_ENVELOPE=0 for the native IMAGE/OCR_RESULT contract.
PADDLE_ENVELOPE = os.environ.get("TRITON_PADDLE_ENVELOPE", "1").lower() not in (
    "0", "false", "no", "off",
)
_DEFAULT_INPUT = "input" if PADDLE_ENVELOPE else "IMAGE"
_DEFAULT_OUTPUT = "output" if PADDLE_ENVELOPE else "OCR_RESULT"

INPUT_NAME = os.environ.get("TRITON_INPUT_NAME", _DEFAULT_INPUT)
OUTPUT_NAME = os.environ.get("TRITON_OUTPUT_NAME", _DEFAULT_OUTPUT)
TEXT_OUTPUT_NAME = os.environ.get("TRITON_TEXT_OUTPUT_NAME", "TEXT")

# PaddleX prunedResult shaping (mirrors compat/paddlex/paddlex_adapter.py so the
# two compat layers stay byte-compatible). Only consulted when PADDLE_ENVELOPE.
REC_SCORE_THRESH = float(os.environ.get("PADDLE_REC_SCORE_THRESH", "0.0"))
USE_TEXTLINE_ORIENTATION = os.environ.get(
    "PADDLE_USE_TEXTLINE_ORIENTATION", "1"
).lower() in ("1", "true", "yes", "on")
EMIT_ORIENTATION_ANGLES = os.environ.get(
    "PADDLEX_EMIT_ORIENTATION_ANGLES", "0"
).lower() in ("1", "true", "yes", "on")
DET_DEFAULTS = {
    "limit_side_len": int(os.environ.get("DET_LIMIT_SIDE_LEN", "64")),
    "limit_type": os.environ.get("DET_LIMIT_TYPE", "min"),
    "thresh": float(os.environ.get("DET_DB_THRESH", "0.2")),
    "box_thresh": float(os.environ.get("DET_BOX_THRESH", "0.45")),
    "unclip_ratio": float(os.environ.get("DET_UNCLIP", "1.4")),
}

MAX_BATCH_SIZE = int(os.environ.get("TRITON_MAX_BATCH_SIZE", "64"))
PDF_DPI = int(os.environ.get("TRITON_PDF_DPI", "100"))
PDF_MODE = os.environ.get("TRITON_PDF_MODE", "auto")

# Pipeline flags forwarded to the backend as query params. Anything outside
# this set is either a Triton protocol reserved word (below) or unsupported.
_PIPELINE_FLAGS = ("layout", "reading_order", "as_blocks", "tables", "formulas", "text")
# Protocol-level parameters a well-behaved Triton client may send. They are
# accepted and ignored rather than rejected: refusing `sequence_id` on a
# stateless model would break clients that set it unconditionally.
_RESERVED_PARAMS = (
    "sequence_id", "sequence_start", "sequence_end", "priority", "timeout",
    "binary_data_output", "classification", "shared_memory_region",
    "shared_memory_byte_size", "shared_memory_offset", "binary_data",
    "binary_data_size",
)

app = FastAPI(title="TurboOCR — Triton/KServe v2 compatible API")


def _client() -> httpx.AsyncClient:
    return app.state.client


@app.on_event("startup")
async def _startup() -> None:
    # One pooled client for the process: a fresh connection per request would
    # dominate latency on a backend that answers in ~30 ms.
    app.state.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await app.state.client.aclose()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
# Triton's error envelope is a bare {"error": "..."} with the status on the
# HTTP response. Clients (tritonclient included) surface `error` verbatim, so
# the message has to carry the whole diagnosis.

def _error(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": msg})


class InferError(Exception):
    """Raised inside request decoding; carries the HTTP status to report."""

    def __init__(self, status: int, msg: str) -> None:
        super().__init__(msg)
        self.status = status
        self.msg = msg


# ---------------------------------------------------------------------------
# BYTES tensor codec (binary tensor data extension)
# ---------------------------------------------------------------------------
# Triton serializes a BYTES tensor as, per element, a 4-byte little-endian
# length followed by that many raw bytes — the same framing numpy object
# arrays get from tritonclient's serialize_byte_tensor(). This is the only
# practical way to send an image: the JSON form of BYTES is a string, which
# cannot hold arbitrary binary.

def _decode_bytes_tensor(buf: bytes, name: str) -> List[bytes]:
    out: List[bytes] = []
    off, end = 0, len(buf)
    while off < end:
        if off + 4 > end:
            raise InferError(400, f"input '{name}': truncated BYTES length prefix at offset {off}")
        (ln,) = struct.unpack_from("<I", buf, off)
        off += 4
        if off + ln > end:
            raise InferError(
                400,
                f"input '{name}': BYTES element claims {ln} bytes but only "
                f"{end - off} remain — is the tensor really BYTES?",
            )
        out.append(bytes(buf[off:off + ln]))
        off += ln
    return out


def _encode_bytes_tensor(items: List[bytes]) -> bytes:
    return b"".join(struct.pack("<I", len(b)) + b for b in items)


def _decode_json_bytes(data: List[Any], name: str) -> List[bytes]:
    """JSON representation of a BYTES tensor.

    Triton's JSON form of BYTES is an array of strings, which cannot carry a
    PNG. Elements are therefore required to be base64 — the same encoding the
    PaddleX-compatible API takes — so `curl` with a JSON body stays usable.

    In PaddleX-envelope mode an element may instead be a JSON object string
    ({"file": <base64>, ...}); such elements are unwrapped by the caller.
    """
    out: List[bytes] = []
    for i, el in enumerate(data):
        if not isinstance(el, str):
            raise InferError(400, f"input '{name}' element {i}: expected a base64 string")
        if PADDLE_ENVELOPE and el.lstrip()[:1] == "{":
            # A JSON object element is a PaddleX input envelope, not base64.
            out.append(_unwrap_envelope(el.encode("utf-8")))
            continue
        try:
            out.append(base64.b64decode(el, validate=False))
        except (binascii.Error, ValueError) as e:
            raise InferError(400, f"input '{name}' element {i}: invalid base64 ({e})") from e
    return out


def _unwrap_envelope(raw: bytes) -> bytes:
    """PaddleX input-envelope compatibility (PADDLE_ENVELOPE mode).

    paddle-ocr-triton clients (e.g. inocr-client) put each element on the wire
    as a JSON object {"file": <base64 image|pdf>, "fileType": 0|1,
    "visualize": bool} rather than the encoded image itself. Unwrap that to the
    decoded file bytes. `fileType` and `visualize` are not needed downstream:
    PDF vs image is sniffed from the decoded bytes, and this backend never
    renders. A non-envelope element (a raw encoded image) is returned
    unchanged, so the native contract keeps working under the same name.
    """
    if raw.lstrip()[:1] != b"{":
        return raw
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(obj, dict) or not isinstance(obj.get("file"), str):
        return raw
    f = obj["file"]
    if f.startswith(("http://", "https://")):
        raise InferError(
            400, f"input '{INPUT_NAME}': URL 'file' is unsupported; send base64-encoded bytes"
        )
    try:
        return base64.b64decode(f, validate=False)
    except (binascii.Error, ValueError) as e:
        raise InferError(
            400, f"input '{INPUT_NAME}': invalid base64 in envelope 'file' ({e})"
        ) from e


def _element_count(shape: Any, name: str) -> int:
    if not isinstance(shape, list) or not shape:
        raise InferError(400, f"input '{name}': 'shape' must be a non-empty array")
    n = 1
    for d in shape:
        if not isinstance(d, int) or d < 0:
            raise InferError(400, f"input '{name}': shape {shape} must hold non-negative integers")
        n *= d
    return n


# ---------------------------------------------------------------------------
# Request decoding
# ---------------------------------------------------------------------------

def _split_body(raw: bytes, header_len: Optional[str]) -> Tuple[bytes, bytes]:
    """Split a possibly-binary request into (JSON header, tensor blob)."""
    if header_len is None:
        return raw, b""
    try:
        n = int(header_len)
    except ValueError as e:
        raise InferError(400, f"Inference-Header-Content-Length is not an integer: {header_len!r}") from e
    if n < 0 or n > len(raw):
        raise InferError(400, f"Inference-Header-Content-Length {n} exceeds the {len(raw)}-byte body")
    return raw[:n], raw[n:]


def _extract_images(payload: Dict[str, Any], blob: bytes) -> List[bytes]:
    """Pull the input tensor's elements out of a decoded infer request.

    Inputs carrying `binary_data_size` are read from the raw blob in the order
    they appear — that ordering is the protocol's, not a convenience.
    """
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise InferError(400, "'inputs' must be a non-empty array")

    images: Optional[List[bytes]] = None
    offset = 0
    for spec in inputs:
        if not isinstance(spec, dict):
            raise InferError(400, "each entry of 'inputs' must be an object")
        name = spec.get("name")
        params = spec.get("parameters") or {}
        size = params.get("binary_data_size")

        if size is not None:
            if not isinstance(size, int) or size < 0:
                raise InferError(400, f"input '{name}': binary_data_size must be a non-negative integer")
            if offset + size > len(blob):
                raise InferError(
                    400,
                    f"input '{name}': binary_data_size {size} runs past the end of the "
                    f"body (have {len(blob) - offset} bytes left)",
                )
            chunk = blob[offset:offset + size]
            offset += size
        else:
            chunk = b""

        # Non-target inputs are consumed for their offset only. A client that
        # sends extra tensors (a leftover from the model it is migrating off)
        # still gets a correctly-framed read of the one we need.
        if name != INPUT_NAME:
            continue

        datatype = spec.get("datatype")
        if datatype != "BYTES":
            raise InferError(
                400,
                f"input '{INPUT_NAME}' has datatype '{datatype}', expected 'BYTES' "
                "(one encoded image per element)",
            )
        if size is not None:
            images = _decode_bytes_tensor(chunk, INPUT_NAME)
            if PADDLE_ENVELOPE:
                # Binary-extension element may be a PaddleX envelope (the shape
                # inocr-client sends via set_data_from_numpy) or a raw image.
                images = [_unwrap_envelope(el) for el in images]
        else:
            data = spec.get("data")
            if not isinstance(data, list):
                raise InferError(
                    400,
                    f"input '{INPUT_NAME}': provide 'data' as an array of base64 strings, "
                    "or use the binary tensor data extension",
                )
            images = _decode_json_bytes(data, INPUT_NAME)

        expected = _element_count(spec.get("shape"), INPUT_NAME)
        if expected != len(images):
            raise InferError(
                400,
                f"input '{INPUT_NAME}': shape {spec.get('shape')} implies {expected} "
                f"elements but {len(images)} were provided",
            )

    if images is None:
        names = [i.get("name") for i in inputs if isinstance(i, dict)]
        raise InferError(400, f"expected an input named '{INPUT_NAME}', got {names}")
    if not images:
        raise InferError(400, f"input '{INPUT_NAME}' is empty")
    if len(images) > MAX_BATCH_SIZE:
        raise InferError(
            400,
            f"batch of {len(images)} exceeds TRITON_MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )
    return images


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        return v.lower() in ("1", "true", "yes", "on")
    return None


def _query_from_parameters(params: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Map KServe request parameters onto TurboOCR query params.

    Returns the query dict plus the names of parameters with no counterpart, so
    the caller can decide between ignoring and rejecting them.
    """
    query: Dict[str, str] = {}
    unsupported: List[str] = []
    for key, value in params.items():
        if key in _PIPELINE_FLAGS:
            b = _as_bool(value)
            if b is None:
                raise InferError(400, f"parameter '{key}' must be a boolean, got {value!r}")
            query[key] = "1" if b else "0"
        elif key in ("pdf_mode", "pdf_dpi"):
            continue  # handled on the PDF path
        elif key in _RESERVED_PARAMS:
            continue
        else:
            unsupported.append(key)
    return query, unsupported


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"%PDF", "application/pdf"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"GIF8", "image/gif"),
)


def _sniff_mime(blob: bytes) -> str:
    """Content-Type from magic bytes.

    Not cosmetic: it is what routes a JPEG to the backend's nvJPEG path and a
    PNG to Wuffs. Sniffing here also avoids a Pillow dependency in this image.
    """
    for magic, mime in _MAGIC:
        if blob.startswith(magic):
            return mime
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _is_pdf(blob: bytes) -> bool:
    return blob.startswith(b"%PDF")


async def _run_one(blob: bytes, query: Dict[str, str], pdf_mode: str, pdf_dpi: int) -> Dict[str, Any]:
    if _is_pdf(blob):
        q = dict(query, mode=pdf_mode, dpi=str(pdf_dpi))
        r = await _client().post(
            f"{TURBO_URL}/ocr/pdf", params=q, content=blob,
            headers={"Content-Type": "application/pdf"},
        )
    else:
        r = await _client().post(
            f"{TURBO_URL}/ocr/raw", params=query, content=blob,
            headers={"Content-Type": _sniff_mime(blob)},
        )
    r.raise_for_status()
    return r.json()


async def _run_batch(blobs: List[bytes], query: Dict[str, str]) -> List[Dict[str, Any]]:
    """Multi-image path.

    Uses the backend's own `/ocr/batch` (nvJPEG batch decode) when every
    element is an image. PDFs have no slot in that endpoint, so a batch
    containing one falls back to sequential single calls rather than silently
    dropping it.
    """
    if any(_is_pdf(b) for b in blobs):
        out = []
        for b in blobs:
            out.append(await _run_one(b, query, PDF_MODE, PDF_DPI))
        return out

    payload = {"images": [base64.b64encode(b).decode() for b in blobs]}
    r = await _client().post(f"{TURBO_URL}/ocr/batch", params=query, json=payload)
    r.raise_for_status()
    body = r.json()
    results = body.get("batch_results", [])
    errors = body.get("errors", [])
    out = []
    for i in range(len(blobs)):
        entry = results[i] if i < len(results) else {}
        # Per-slot failures stay in-band and index-aligned: a Triton response
        # tensor must have one element per input, so a bad slot cannot be
        # dropped without desynchronising the caller's batch.
        if i < len(errors) and errors[i]:
            entry = dict(entry or {})
            entry["error"] = errors[i]
        out.append(entry or {})
    return out


def _flatten_text(doc: Dict[str, Any]) -> str:
    """Recognized text for the TEXT output tensor.

    Lines are newline-joined; PDF pages are separated by a form feed, matching
    the convention pdftotext established.
    """
    if "pages" in doc:
        pages = doc.get("pages") or []
        return "\f".join(
            "\n".join(str(r.get("text", "")) for r in (p.get("results") or []))
            for p in pages
        )
    return "\n".join(str(r.get("text", "")) for r in (doc.get("results") or []))


# ---------------------------------------------------------------------------
# Response encoding
# ---------------------------------------------------------------------------

def _wants_binary(payload: Dict[str, Any], spec: Optional[Dict[str, Any]], request_was_binary: bool) -> bool:
    """Resolve the binary_data setting for one output.

    Explicit per-output wins, then the request-level `binary_data_output`,
    then the shape of the request itself. That last fallback is deliberate:
    tritonclient always sends binary and always sets the flag, while a human
    with curl sends JSON and wants JSON back.
    """
    if spec is not None:
        v = (spec.get("parameters") or {}).get("binary_data")
        b = _as_bool(v)
        if b is not None:
            return b
    v = (payload.get("parameters") or {}).get("binary_data_output")
    b = _as_bool(v)
    if b is not None:
        return b
    return request_was_binary


def _requested_outputs(payload: Dict[str, Any]) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Outputs to emit, in the order the client asked for them.

    A request with no `outputs` gets every output, which is what Triton does.
    """
    outs = payload.get("outputs")
    if not isinstance(outs, list) or not outs:
        return [(OUTPUT_NAME, None), (TEXT_OUTPUT_NAME, None)]
    resolved: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for spec in outs:
        if not isinstance(spec, dict):
            raise InferError(400, "each entry of 'outputs' must be an object")
        name = spec.get("name")
        if name not in (OUTPUT_NAME, TEXT_OUTPUT_NAME):
            raise InferError(
                400,
                f"unexpected output '{name}'; this model produces "
                f"'{OUTPUT_NAME}' and '{TEXT_OUTPUT_NAME}'",
            )
        resolved.append((name, spec))
    return resolved


def _poly_to_box(poly: List[List[float]]) -> List[int]:
    """Axis-aligned [x1,y1,x2,y2] enclosing a 4-point polygon (PaddleX rec_boxes)."""
    xs = [int(round(p[0])) for p in poly]
    ys = [int(round(p[1])) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _pruned_result(turbo_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map TurboOCR `results[]` onto PaddleX's prunedResult.

    Field set and key order mirror compat/paddlex/paddlex_adapter.py so a
    paddle-ocr-triton client parses either backend identically. dt_polys is
    everything detected; rec_* is what survived REC_SCORE_THRESH.
    """
    dt_polys: List[List[List[int]]] = []
    rec_texts: List[str] = []
    rec_scores: List[float] = []
    rec_polys: List[List[List[int]]] = []
    rec_boxes: List[List[int]] = []
    for item in turbo_results or []:
        poly = [[int(round(c[0])), int(round(c[1]))] for c in item.get("bounding_box", [])]
        if not poly:
            continue
        dt_polys.append(poly)
        score = float(item.get("confidence", 0.0))
        if score < REC_SCORE_THRESH:
            continue
        rec_texts.append(item.get("text", ""))
        rec_scores.append(score)
        rec_polys.append(poly)
        rec_boxes.append(_poly_to_box(poly))
    pruned: Dict[str, Any] = {
        "model_settings": {
            "use_doc_preprocessor": False,
            "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        },
        "dt_polys": dt_polys,
        "text_det_params": dict(DET_DEFAULTS),
        "text_type": "general",
    }
    if EMIT_ORIENTATION_ANGLES:
        pruned["textline_orientation_angles"] = [0] * len(dt_polys)
    pruned["text_rec_score_thresh"] = REC_SCORE_THRESH
    pruned["return_word_box"] = False
    pruned["rec_texts"] = rec_texts
    pruned["rec_scores"] = rec_scores
    pruned["rec_polys"] = rec_polys
    pruned["rec_boxes"] = rec_boxes
    return pruned


def _ocr_entry(pruned: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prunedResult": pruned,
        "ocrImage": None,
        "docPreprocessingImage": None,
        "inputImage": None,
    }


def _to_paddle_envelope(doc: Dict[str, Any]) -> Dict[str, Any]:
    """One TurboOCR result doc -> the PaddleX-HPS response envelope that
    paddle-ocr-triton clients (inocr-client) parse.

    A per-slot backend failure (`error` in a batch entry) maps to a non-zero
    errorCode so the client raises, rather than silently returning empty text.
    """
    err = doc.get("error")
    if err:
        return {"logId": "", "errorCode": 1, "errorMsg": str(err),
                "result": {"ocrResults": [], "dataInfo": {}}}
    if "pages" in doc:
        pages = doc.get("pages") or []
        ocr_results = [_ocr_entry(_pruned_result(p.get("results", []))) for p in pages]
        data_info: Dict[str, Any] = {
            "numPages": len(pages),
            "pages": [{"width": int(p.get("width", 0)), "height": int(p.get("height", 0))}
                      for p in pages],
            "type": "pdf",
        }
    else:
        ocr_results = [_ocr_entry(_pruned_result(doc.get("results", [])))]
        data_info = {"width": int(doc.get("width", 0)), "height": int(doc.get("height", 0)),
                     "type": "image"}
    return {"logId": "", "errorCode": 0, "errorMsg": "Success",
            "result": {"ocrResults": ocr_results, "dataInfo": data_info}}


def _build_response(
    payload: Dict[str, Any],
    docs: List[Dict[str, Any]],
    request_was_binary: bool,
) -> Response:
    header: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "outputs": [],
    }
    if payload.get("id") is not None:
        header["id"] = payload["id"]

    blobs: List[bytes] = []
    for name, spec in _requested_outputs(payload):
        if name == OUTPUT_NAME:
            if PADDLE_ENVELOPE:
                elements = [json.dumps(_to_paddle_envelope(d), separators=(",", ":")) for d in docs]
            else:
                elements = [json.dumps(d, separators=(",", ":")) for d in docs]
        else:
            elements = [_flatten_text(d) for d in docs]

        out: Dict[str, Any] = {"name": name, "datatype": "BYTES", "shape": [len(elements)]}
        if _wants_binary(payload, spec, request_was_binary):
            blob = _encode_bytes_tensor([e.encode() for e in elements])
            out["parameters"] = {"binary_data_size": len(blob)}
            blobs.append(blob)
        else:
            # Triton's JSON form of a BYTES tensor is an array of plain
            # strings, so both outputs go out as text — NOT base64. The
            # asymmetry with the input side is forced: an input element is an
            # encoded image and has no string form, while both outputs are
            # UTF-8 by construction (JSON, and dictionary-decoded text).
            out["data"] = elements
        header["outputs"].append(out)

    if not blobs:
        return JSONResponse(status_code=200, content=header)

    head = json.dumps(header, separators=(",", ":")).encode()
    return Response(
        status_code=200,
        content=head + b"".join(blobs),
        media_type="application/octet-stream",
        headers={"Inference-Header-Content-Length": str(len(head))},
    )


# ---------------------------------------------------------------------------
# Server / model metadata
# ---------------------------------------------------------------------------

@app.get("/v2")
async def server_metadata() -> Dict[str, Any]:
    return {
        "name": "turboocr",
        "version": SERVER_VERSION,
        # Advertise only what is actually implemented. A client that probes
        # for shared memory and finds it listed but absent fails much later
        # and much more confusingly than one told the truth here.
        "extensions": ["binary_tensor_data", "model_repository", "model_configuration"],
    }


@app.get("/v2/health/live")
async def health_live() -> Response:
    return Response(status_code=200)


async def _backend_ready() -> bool:
    try:
        r = await _client().get(f"{TURBO_URL}/health/ready", timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — any failure is "not ready"
        return False


@app.get("/v2/health/ready")
async def health_ready() -> Response:
    # Reports the backend's readiness, not the adapter's: an adapter that
    # answers 200 while the engines are still building would let Kubernetes
    # route traffic into a server that cannot serve it.
    if await _backend_ready():
        return Response(status_code=200)
    return _error(400, "backend not ready")


def _check_model(name: str, version: Optional[str] = None) -> Optional[JSONResponse]:
    if name != MODEL_NAME:
        return _error(404, f"Request for unknown model: '{name}' is not found")
    if version is not None and version != MODEL_VERSION:
        return _error(404, f"Request for unknown version: '{version}' is not found for model '{name}'")
    return None


@app.get("/v2/models/{model}/ready", response_model=None)
@app.get("/v2/models/{model}/versions/{version}/ready", response_model=None)
async def model_ready(model: str, version: Optional[str] = None) -> Response:
    bad = _check_model(model, version)
    if bad is not None:
        return bad
    if await _backend_ready():
        return Response(status_code=200)
    return _error(400, "backend not ready")


def _model_metadata() -> Dict[str, Any]:
    return {
        "name": MODEL_NAME,
        "versions": [MODEL_VERSION],
        "platform": "turboocr",
        "inputs": [{"name": INPUT_NAME, "datatype": "BYTES", "shape": [-1]}],
        "outputs": [
            {"name": OUTPUT_NAME, "datatype": "BYTES", "shape": [-1]},
            {"name": TEXT_OUTPUT_NAME, "datatype": "BYTES", "shape": [-1]},
        ],
    }


@app.get("/v2/models/{model}", response_model=None)
@app.get("/v2/models/{model}/versions/{version}", response_model=None)
async def model_metadata(model: str, version: Optional[str] = None) -> Union[JSONResponse, Dict[str, Any]]:
    bad = _check_model(model, version)
    if bad is not None:
        return bad
    return _model_metadata()


@app.get("/v2/models/{model}/config", response_model=None)
@app.get("/v2/models/{model}/versions/{version}/config", response_model=None)
async def model_config(model: str, version: Optional[str] = None) -> Union[JSONResponse, Dict[str, Any]]:
    bad = _check_model(model, version)
    if bad is not None:
        return bad
    # max_batch_size 0 declares "no dynamic batching, shapes are as given" —
    # honest here, because batching happens inside the C++ pipeline pool and
    # is not something a Triton scheduler is arranging.
    return {
        "name": MODEL_NAME,
        "platform": "turboocr",
        "backend": "turboocr",
        "max_batch_size": 0,
        "input": [{"name": INPUT_NAME, "data_type": "TYPE_STRING", "dims": [-1]}],
        "output": [
            {"name": OUTPUT_NAME, "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": TEXT_OUTPUT_NAME, "data_type": "TYPE_STRING", "dims": [-1]},
        ],
    }


@app.post("/v2/repository/index", response_model=None)
async def repository_index() -> List[Dict[str, Any]]:
    ready = await _backend_ready()
    return [{
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "state": "READY" if ready else "UNAVAILABLE",
        "reason": "" if ready else "backend not ready",
    }]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@app.post("/v2/models/{model}/infer", response_model=None)
@app.post("/v2/models/{model}/versions/{version}/infer", response_model=None)
async def infer(model: str, request: Request, version: Optional[str] = None) -> Response:
    bad = _check_model(model, version)
    if bad is not None:
        return bad

    raw = await request.body()
    header_len = request.headers.get("Inference-Header-Content-Length")
    request_was_binary = header_len is not None

    try:
        head, blob = _split_body(raw, header_len)
        try:
            payload = json.loads(head)
        except json.JSONDecodeError as e:
            raise InferError(400, f"request header is not valid JSON: {e}") from e
        if not isinstance(payload, dict):
            raise InferError(400, "request header must be a JSON object")

        images = _extract_images(payload, blob)

        params = payload.get("parameters") or {}
        if not isinstance(params, dict):
            raise InferError(400, "'parameters' must be an object")
        query, unsupported = _query_from_parameters(params)
        if unsupported and STRICT_PARAMS:
            raise InferError(
                400,
                "parameters not supported by the TurboOCR backend: "
                + ", ".join(sorted(unsupported))
                + ". See TRITON_COMPAT.md; unset TRITON_STRICT_PARAMS to ignore them.",
            )
        if unsupported:
            logger.info("ignoring unsupported parameters: %s", sorted(unsupported))

        pdf_mode = str(params.get("pdf_mode", PDF_MODE))
        try:
            pdf_dpi = int(params.get("pdf_dpi", PDF_DPI))
        except (TypeError, ValueError) as e:
            raise InferError(400, f"parameter 'pdf_dpi' must be an integer: {e}") from e

        # Validate the requested outputs before doing any GPU work.
        _requested_outputs(payload)
    except InferError as e:
        return _error(e.status, e.msg)

    try:
        if len(images) == 1:
            docs = [await _run_one(images[0], query, pdf_mode, pdf_dpi)]
        else:
            docs = await _run_batch(images, query)
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500] if e.response is not None else str(e)
        logger.error("backend rejected request: %s", detail)
        # The backend's own 4xx (LAYOUT_DISABLED, TABLE_BACKEND_DISABLED, …) is
        # the caller's fault and must not be laundered into a 500.
        status = e.response.status_code if e.response is not None else 500
        return _error(status if 400 <= status < 500 else 500, f"backend error: {detail}")
    except httpx.HTTPError as e:
        logger.error("backend unreachable: %s", e)
        return _error(503, f"backend unreachable: {e}")
    except Exception as e:  # noqa: BLE001
        logger.exception("unhandled failure")
        return _error(500, f"internal error: {e}")

    try:
        return _build_response(payload, docs, request_was_binary)
    except InferError as e:
        return _error(e.status, e.msg)
