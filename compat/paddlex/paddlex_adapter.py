"""PaddleX-compatible OCR API in front of TurboOCR.

Speaks the PaddleX 3.x pipeline-serving contract (`POST /ocr`, `GET /health`)
and translates each request into TurboOCR's native HTTP API, so an existing
PaddleX client can point at this process unchanged.

Schema is taken from PaddleX 3.7.2:
  paddlex/inference/serving/schemas/ocr.py            request + result models
  paddlex/inference/serving/schemas/shared/ocr.py     file / fileType
  paddlex/inference/serving/infra/models.py           response envelope, dataInfo
  paddlex/inference/pipelines/ocr/result.py           prunedResult key order
  .../_pipeline_apps/_common/common.py                prune_result()

Not every PaddleX knob has a TurboOCR equivalent — several are server-level
rather than per-request, and a few have no counterpart at all. See
PADDLEX_COMPAT.md for the field-by-field differences. Unsupported parameters
are ignored by default; set PADDLEX_STRICT_PARAMS=1 to reject them with 422
instead of silently doing something different from what the caller asked.

Run:
    uvicorn paddlex_adapter:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("paddlex_adapter")

TURBO_URL = os.environ.get("TURBO_OCR_URL", "http://127.0.0.1:8081").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("PADDLEX_TIMEOUT_S", "120"))
STRICT_PARAMS = os.environ.get("PADDLEX_STRICT_PARAMS", "0").lower() in ("1", "true", "yes", "on")
MAX_NUM_INPUT_IMGS = int(os.environ.get("MAX_NUM_INPUT_IMGS", "1000"))
PDF_DPI = int(os.environ.get("PADDLEX_PDF_DPI", "100"))
# TurboOCR does not report a per-line 0/180 decision, so the optional
# `textline_orientation_angles` key is omitted rather than filled with invented
# values. Clients that require the key to exist can opt into zeros.
EMIT_ORIENTATION_ANGLES = os.environ.get("PADDLEX_EMIT_ORIENTATION_ANGLES", "0").lower() in ("1", "true", "yes", "on")

# Mirrors the detection defaults the server reports at startup; surfaced back to
# the caller inside prunedResult.text_det_params so the echoed values are the
# ones actually in force, not the ones requested.
DET_DEFAULTS = {
    "limit_side_len": int(os.environ.get("DET_LIMIT_SIDE_LEN", "64")),
    "limit_type": os.environ.get("DET_LIMIT_TYPE", "min"),
    "thresh": float(os.environ.get("DET_DB_THRESH", "0.2")),
    "box_thresh": float(os.environ.get("DET_BOX_THRESH", "0.45")),
    "unclip_ratio": float(os.environ.get("DET_UNCLIP", "1.4")),
}

# Per-request knobs PaddleX accepts that TurboOCR can only honour process-wide
# (they are read from env at server start and baked into TRT engines).
_SERVER_LEVEL_PARAMS = (
    "textDetLimitSideLen", "textDetLimitType", "textDetThresh",
    "textDetBoxThresh", "textDetUnclipRatio", "useTextlineOrientation",
    "useDocOrientationClassify",
)
# No TurboOCR counterpart at all.
_UNSUPPORTED_PARAMS = ("useDocUnwarping", "returnWordBox", "visualize")


class InferRequest(BaseModel):
    """PaddleX 3.x OCR InferRequest."""

    file: str
    fileType: Optional[Literal[0, 1]] = None
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = None
    textDetLimitSideLen: Optional[int] = None
    textDetLimitType: Optional[str] = None
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textRecScoreThresh: Optional[float] = None
    returnWordBox: Optional[bool] = None
    visualize: Optional[bool] = None
    logId: Optional[str] = None


app = FastAPI(title="TurboOCR — PaddleX-compatible API")


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


def _log_id(supplied: Optional[str] = None) -> str:
    return supplied or str(uuid.uuid4())


def _error(status: int, msg: str, log_id: Optional[str] = None) -> JSONResponse:
    """PaddleX error envelope: errorCode mirrors the HTTP status."""
    return JSONResponse(
        status_code=status,
        content={"logId": _log_id(log_id), "errorCode": status, "errorMsg": msg},
    )


@app.exception_handler(ValidationError)
async def _validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return _error(422, str(exc))


@app.get("/health", response_model=None)
async def health() -> Union[JSONResponse, Dict[str, Any]]:
    """PaddleX health probe. Reports the backend's readiness, not just liveness."""
    log_id = _log_id()
    try:
        r = await _client().get(f"{TURBO_URL}/health/ready", timeout=5)
        if r.status_code == 200:
            return {"logId": log_id, "errorCode": 0, "errorMsg": "Healthy"}
        msg = f"backend not ready (HTTP {r.status_code})"
    except Exception as e:  # noqa: BLE001 — surfaced to the caller verbatim
        msg = f"backend unreachable: {e}"
    return JSONResponse(
        status_code=503, content={"logId": log_id, "errorCode": 503, "errorMsg": msg}
    )


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _infer_file_type(url: str) -> Optional[int]:
    """0 = PDF, 1 = image — matching PaddleX's extension sniff."""
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    if path.endswith(".pdf"):
        return 0
    if path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
        return 1
    return None


async def _fetch_bytes(file: str) -> bytes:
    if _is_url(file):
        r = await _client().get(file)
        r.raise_for_status()
        return r.content
    # PaddleX uses a plain b64decode here; keep validate=False so the same
    # lenient inputs (whitespace, newlines from wrapped base64) still work.
    return base64.b64decode(file)


def _resolve_file_type(req: InferRequest) -> int:
    if req.fileType is not None:
        return req.fileType
    if _is_url(req.file):
        inferred = _infer_file_type(req.file)
        if inferred is None:
            raise ValueError("Unsupported file type")
        return inferred
    raise ValueError("File type cannot be determined")


def _poly_to_box(poly: List[List[float]]) -> List[int]:
    """Axis-aligned [x1,y1,x2,y2] enclosing a 4-point polygon (PaddleX rec_boxes)."""
    xs = [int(round(p[0])) for p in poly]
    ys = [int(round(p[1])) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _build_pruned_result(
    turbo_results: List[Dict[str, Any]],
    rec_score_thresh: float,
    use_textline_orientation: bool,
) -> Dict[str, Any]:
    """Map TurboOCR `results[]` onto PaddleX's prunedResult.

    Key order follows OCRResult._to_json() in PaddleX. `input_path` and
    `page_index` are deliberately absent: prune_result() strips them before
    serialization, so a compatible payload must not carry them.

    PaddleX distinguishes dt_polys (everything detected) from rec_polys (what
    survived text_rec_score_thresh); that distinction is reproduced here by
    filtering on confidence.
    """
    dt_polys: List[List[List[int]]] = []
    rec_texts: List[str] = []
    rec_scores: List[float] = []
    rec_polys: List[List[List[int]]] = []
    rec_boxes: List[List[int]] = []

    for item in turbo_results:
        poly = [[int(round(c[0])), int(round(c[1]))] for c in item.get("bounding_box", [])]
        if not poly:
            continue
        dt_polys.append(poly)
        score = float(item.get("confidence", 0.0))
        if score < rec_score_thresh:
            continue
        rec_texts.append(item.get("text", ""))
        rec_scores.append(score)
        rec_polys.append(poly)
        rec_boxes.append(_poly_to_box(poly))

    pruned: Dict[str, Any] = {
        "model_settings": {
            # No doc-preprocessor stage exists here, so doc_preprocessor_res is
            # correspondingly absent — consistent with how PaddleX omits it.
            "use_doc_preprocessor": False,
            "use_textline_orientation": use_textline_orientation,
        },
        "dt_polys": dt_polys,
        "text_det_params": dict(DET_DEFAULTS),
        "text_type": "general",
    }
    if EMIT_ORIENTATION_ANGLES:
        pruned["textline_orientation_angles"] = [0] * len(dt_polys)
    pruned["text_rec_score_thresh"] = rec_score_thresh
    pruned["return_word_box"] = False
    pruned["rec_texts"] = rec_texts
    pruned["rec_scores"] = rec_scores
    pruned["rec_polys"] = rec_polys
    pruned["rec_boxes"] = rec_boxes
    return pruned


def _ocr_result_entry(pruned: Dict[str, Any]) -> Dict[str, Any]:
    # Image fields are always null: rendering an annotated image would mean
    # decoding and re-encoding every page, which defeats the point of this
    # backend. `visualize` is documented as unsupported.
    return {
        "prunedResult": pruned,
        "ocrImage": None,
        "docPreprocessingImage": None,
        "inputImage": None,
    }


def _unsupported_in_request(req: InferRequest) -> List[str]:
    named = [p for p in _SERVER_LEVEL_PARAMS + _UNSUPPORTED_PARAMS
             if getattr(req, p, None) is not None]
    # visualize=False asks for nothing, so it is always satisfiable.
    if req.visualize is False and "visualize" in named:
        named.remove("visualize")
    return named


_PIL_TO_MIME = {
    "PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
    "BMP": "image/bmp", "TIFF": "image/tiff",
}


async def _run_image(body: bytes) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    # Decoding here serves two purposes: dataInfo needs the dimensions, and the
    # detected format picks the Content-Type that routes the request to
    # TurboOCR's fast decode path (nvJPEG for JPEG, Wuffs for PNG).
    with Image.open(io.BytesIO(body)) as im:
        size = (im.width, im.height)
        mime = _PIL_TO_MIME.get(im.format or "", "application/octet-stream")
    r = await _client().post(
        f"{TURBO_URL}/ocr/raw", content=body, headers={"Content-Type": mime}
    )
    r.raise_for_status()
    return r.json().get("results", []), size


async def _run_pdf(body: bytes) -> List[Dict[str, Any]]:
    r = await _client().post(
        f"{TURBO_URL}/ocr/pdf?mode=auto&dpi={PDF_DPI}",
        content=body,
        headers={"Content-Type": "application/pdf"},
    )
    r.raise_for_status()
    return r.json().get("pages", [])


# response_model=None: the handler returns either a plain dict (success) or a
# JSONResponse (error envelope), and FastAPI cannot derive a response model
# from that union.
@app.post("/ocr", response_model=None)
async def ocr(request: Request) -> Union[JSONResponse, Dict[str, Any]]:
    try:
        payload = await request.json()
    except Exception:
        return _error(422, "Request body is not valid JSON")

    try:
        req = InferRequest(**payload)
    except ValidationError as e:
        return _error(422, str(e))

    log_id = _log_id(req.logId)

    if STRICT_PARAMS:
        unsupported = _unsupported_in_request(req)
        if unsupported:
            return _error(
                422,
                "Parameters not supported by the TurboOCR backend: "
                + ", ".join(sorted(unsupported))
                + ". See PADDLEX_COMPAT.md; unset PADDLEX_STRICT_PARAMS to ignore them.",
                log_id,
            )

    try:
        file_type = _resolve_file_type(req)
    except ValueError as e:
        return _error(422, str(e), log_id)

    try:
        body = await _fetch_bytes(req.file)
    except (binascii.Error, ValueError):
        return _error(422, "Invalid input file", log_id)
    except httpx.HTTPError as e:
        return _error(422, f"Invalid input file: {e}", log_id)

    if not body:
        return _error(422, "Invalid input file", log_id)

    rec_thresh = req.textRecScoreThresh if req.textRecScoreThresh is not None else 0.0
    use_tlo = req.useTextlineOrientation if req.useTextlineOrientation is not None else True

    try:
        if file_type == 0:
            pages = await _run_pdf(body)
            if len(pages) > MAX_NUM_INPUT_IMGS:
                return _error(
                    422,
                    f"Too many pages: {len(pages)} > MAX_NUM_INPUT_IMGS={MAX_NUM_INPUT_IMGS}",
                    log_id,
                )
            ocr_results = [
                _ocr_result_entry(
                    _build_pruned_result(p.get("results", []), rec_thresh, use_tlo)
                )
                for p in pages
            ]
            data_info: Dict[str, Any] = {
                "numPages": len(pages),
                "pages": [
                    {"width": int(p.get("width", 0)), "height": int(p.get("height", 0))}
                    for p in pages
                ],
                "type": "pdf",
            }
        else:
            results, (w, h) = await _run_image(body)
            ocr_results = [
                _ocr_result_entry(_build_pruned_result(results, rec_thresh, use_tlo))
            ]
            data_info = {"width": w, "height": h, "type": "image"}
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500] if e.response is not None else str(e)
        logger.error("backend rejected request: %s", detail)
        return _error(500, f"Backend error: {detail}", log_id)
    except httpx.HTTPError as e:
        logger.error("backend unreachable: %s", e)
        return _error(500, f"Backend unreachable: {e}", log_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("unhandled failure")
        return _error(500, f"Internal error: {e}", log_id)

    return {
        "logId": log_id,
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {"ocrResults": ocr_results, "dataInfo": data_info},
    }
