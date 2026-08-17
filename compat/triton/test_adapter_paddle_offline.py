"""PaddleX-envelope (default) mode checks for the Triton adapter.

The adapter defaults to a drop-in for the paddle-ocr-triton `ocr` model: input
tensor `input` carrying a PaddleX envelope {"file": <base64>, "fileType", ...},
output tensor `output` carrying the PaddleX-HPS JSON envelope. This verifies an
UNMODIFIED inocr-client works by only repointing its endpoint, against a
stubbed backend (no GPU, no model, no live turboocr-server).

    python3 compat/triton/test_adapter_paddle_offline.py
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys

import httpx
from fastapi.testclient import TestClient

os.environ.pop("TRITON_PADDLE_ENVELOPE", None)  # default is PaddleX mode

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import triton_adapter as ta  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
PDF = b"%PDF-1.4 fake"

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def _doc(text):
    return {"results": [{"text": text, "confidence": 0.9,
                         "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]]}],
            "width": 10, "height": 5}


seen = {}


def backend(request: httpx.Request) -> httpx.Response:
    seen["path"] = request.url.path
    if request.url.path == "/health/ready":
        return httpx.Response(200, text="ready")
    if request.url.path == "/ocr/raw":
        seen["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json=_doc("hello"))
    if request.url.path == "/ocr/pdf":
        return httpx.Response(200, json={"pages": [
            {"page_number": 1, "width": 8, "height": 4, "results": [{"text": "p1",
                "confidence": 0.8, "bounding_box": [[0, 0], [8, 0], [8, 4], [0, 4]]}]},
            {"page_number": 2, "width": 8, "height": 4, "results": [{"text": "p2",
                "confidence": 0.7, "bounding_box": [[0, 0], [8, 0], [8, 4], [0, 4]]}]},
        ]})
    return httpx.Response(404, json={"error": "no stub"})


def encode_bytes(items):
    return b"".join(struct.pack("<I", len(b)) + b for b in items)


def decode_bytes(buf):
    out, off = [], 0
    while off < len(buf):
        (ln,) = struct.unpack_from("<I", buf, off)
        off += 4
        out.append(buf[off:off + ln])
        off += ln
    return out


def envelope(blob, file_type=1):
    """The exact element inocr-client puts on the wire."""
    return json.dumps({"file": base64.b64encode(blob).decode("ascii"),
                       "visualize": False, "fileType": file_type})


def infer_binary(client, element_str, shape):
    """Post the way inocr-client does: input `input`, BYTES, binary extension."""
    blob = encode_bytes([element_str.encode("utf-8")])
    header = {"inputs": [{
        "name": "input", "datatype": "BYTES", "shape": shape,
        "parameters": {"binary_data_size": len(blob)},
    }]}
    head = json.dumps(header).encode()
    return client.post(
        "/v2/models/ocr/infer",
        content=head + blob,
        headers={"Inference-Header-Content-Length": str(len(head)),
                 "Content-Type": "application/octet-stream"},
    )


def read_output(r, name="output"):
    """Pull one BYTES output element back out (binary or json response)."""
    ct = r.headers.get("content-type", "")
    if ct.startswith("application/json"):
        j = r.json()
        spec = next(o for o in j["outputs"] if o["name"] == name)
        return spec["data"][0]
    hl = int(r.headers["Inference-Header-Content-Length"])
    head = json.loads(r.content[:hl])
    body = r.content[hl:]
    off = 0
    for o in head["outputs"]:
        sz = o["parameters"]["binary_data_size"]
        if o["name"] == name:
            return decode_bytes(body[off:off + sz])[0]
        off += sz
    raise KeyError(name)


with TestClient(ta.app) as c:
    ta.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))

    print("metadata advertises the paddle contract")
    j = c.get("/v2/models/ocr").json()
    check("input named 'input'", j["inputs"][0]["name"] == "input", str(j["inputs"]))
    check("output named 'output'", j["outputs"][0]["name"] == "output",
          str([o["name"] for o in j["outputs"]]))
    cfg = c.get("/v2/models/ocr/config").json()
    check("config input named 'input'", cfg["input"][0]["name"] == "input")

    print("\ninocr-client shape: envelope in, PaddleX envelope out")
    r = infer_binary(c, envelope(PNG, 1), shape=[1, 1])
    check("200", r.status_code == 200, r.text[:200])
    check("routed to /ocr/raw", seen["path"] == "/ocr/raw", seen.get("path"))
    check("png sniffed from decoded bytes", seen.get("content_type") == "image/png",
          str(seen.get("content_type")))
    env = json.loads(read_output(r, "output"))
    check("errorCode 0", env.get("errorCode") == 0, str(env)[:200])
    check("has result.ocrResults", isinstance(env.get("result", {}).get("ocrResults"), list)
          and len(env["result"]["ocrResults"]) == 1)
    pr = env["result"]["ocrResults"][0]["prunedResult"]
    check("rec_texts", pr.get("rec_texts") == ["hello"], str(pr.get("rec_texts")))
    check("rec_scores parallel", len(pr["rec_scores"]) == 1 and abs(pr["rec_scores"][0] - 0.9) < 1e-6)
    check("dt_polys 4-point", len(pr["dt_polys"]) == 1 and len(pr["dt_polys"][0]) == 4)
    check("rec_polys present", pr.get("rec_polys") == pr["dt_polys"])
    check("rec_boxes axis-aligned", pr.get("rec_boxes") == [[0, 0, 10, 5]], str(pr.get("rec_boxes")))
    check("dataInfo type image", env["result"]["dataInfo"].get("type") == "image")

    print("\nthe convenience TEXT output still works")
    r = infer_binary(c, envelope(PNG, 1), shape=[1, 1])
    check("TEXT is plain", read_output(r, "TEXT") == b"hello", read_output(r, "TEXT")[:40])

    print("\nbackward compat: a raw encoded image (no envelope) still works")
    blob = encode_bytes([PNG])
    header = {"inputs": [{"name": "input", "datatype": "BYTES", "shape": [1],
                          "parameters": {"binary_data_size": len(blob)}}]}
    head = json.dumps(header).encode()
    r = c.post("/v2/models/ocr/infer", content=head + blob,
               headers={"Inference-Header-Content-Length": str(len(head)),
                        "Content-Type": "application/octet-stream"})
    check("raw image 200", r.status_code == 200, r.text[:200])
    env = json.loads(read_output(r, "output"))
    check("raw image rec_texts", env["result"]["ocrResults"][0]["prunedResult"]["rec_texts"] == ["hello"])

    print("\nJSON-form envelope (curl-friendly) works too")
    r = c.post("/v2/models/ocr/infer", json={
        "inputs": [{"name": "input", "datatype": "BYTES", "shape": [1, 1],
                    "data": [envelope(PNG, 1)]}],
        "outputs": [{"name": "output"}]})
    check("json-form 200", r.status_code == 200, r.text[:200])
    env = json.loads(r.json()["outputs"][0]["data"][0])
    check("json-form rec_texts", env["result"]["ocrResults"][0]["prunedResult"]["rec_texts"] == ["hello"])

    print("\nPDF envelope (fileType 0) -> per-page ocrResults")
    r = infer_binary(c, envelope(PDF, 0), shape=[1, 1])
    check("routed to /ocr/pdf", seen["path"] == "/ocr/pdf", seen.get("path"))
    env = json.loads(read_output(r, "output"))
    check("two pages", len(env["result"]["ocrResults"]) == 2)
    check("numPages", env["result"]["dataInfo"].get("numPages") == 2)
    check("page 1 text", env["result"]["ocrResults"][0]["prunedResult"]["rec_texts"] == ["p1"])
    check("page 2 text", env["result"]["ocrResults"][1]["prunedResult"]["rec_texts"] == ["p2"])

    print("\nerror surfacing: bad base64 in envelope 'file'")
    r = c.post("/v2/models/ocr/infer", json={
        "inputs": [{"name": "input", "datatype": "BYTES", "shape": [1, 1],
                    "data": [json.dumps({"file": "!!!not-base64!!!", "fileType": 1})]}]})
    # padding-tolerant decoders may accept it; either a 400 here or a backend
    # decode path is acceptable — what must NOT happen is a 500/crash.
    check("bad-base64 handled cleanly", r.status_code in (200, 400, 422), str(r.status_code))

print()
if fails:
    print(f"FAILED ({len(fails)}): {', '.join(fails)}")
    sys.exit(1)
print("all paddle-mode checks passed")
