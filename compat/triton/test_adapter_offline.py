"""Protocol-level checks for the Triton adapter against a stubbed backend.

Runs without a GPU, a model, or a live turboocr-server: the backend is an
httpx MockTransport, so what is under test is purely the KServe v2 wire
format — binary tensor framing, batch alignment, metadata, error mapping.

    python3 compat/triton/test_adapter_offline.py

For an end-to-end check against a real server, use test_adapter_live.py.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys

import httpx
from fastapi.testclient import TestClient

# This file validates the NATIVE tensor contract (IMAGE / OCR_RESULT + TEXT).
# The adapter now defaults to PaddleX-envelope mode, so pin native mode before
# import (the module reads these at import time). PaddleX mode is covered by
# test_adapter_paddle_offline.py.
os.environ["TRITON_PADDLE_ENVELOPE"] = "0"

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
                         "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]]}]}


seen = {}


def backend(request: httpx.Request) -> httpx.Response:
    seen["path"] = request.url.path
    seen["query"] = dict(request.url.params)
    if request.url.path == "/health/ready":
        return httpx.Response(200, text="ready")
    if request.url.path == "/ocr/raw":
        seen["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json=_doc("hello"))
    if request.url.path == "/ocr/pdf":
        return httpx.Response(200, json={"pages": [
            {"page_number": 1, "results": [{"text": "p1"}]},
            {"page_number": 2, "results": [{"text": "p2"}]},
        ]})
    if request.url.path == "/ocr/batch":
        body = json.loads(request.content)
        n = len(body["images"])
        seen["batch_n"] = n
        return httpx.Response(200, json={
            "batch_results": [_doc(f"img{i}") for i in range(n)],
            "errors": [None] * (n - 1) + ["decode_failed"],
        })
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


def binary_infer(client, images, params=None, outputs=None):
    """Post an infer request the way tritonclient.http does."""
    blob = encode_bytes(images)
    header = {
        "inputs": [{
            "name": "IMAGE", "datatype": "BYTES", "shape": [len(images)],
            "parameters": {"binary_data_size": len(blob)},
        }],
    }
    if params:
        header["parameters"] = params
    if outputs is not None:
        header["outputs"] = outputs
    head = json.dumps(header).encode()
    return client.post(
        "/v2/models/ocr/infer",
        content=head + blob,
        headers={"Inference-Header-Content-Length": str(len(head)),
                 "Content-Type": "application/octet-stream"},
    )


with TestClient(ta.app) as c:
    # Replace the pooled backend client created at startup with the stub.
    ta.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))

    print("metadata")
    r = c.get("/v2")
    check("GET /v2", r.status_code == 200 and "binary_tensor_data" in r.json()["extensions"])
    check("live", c.get("/v2/health/live").status_code == 200)
    check("ready", c.get("/v2/health/ready").status_code == 200)
    r = c.get("/v2/models/ocr")
    j = r.json()
    check("model metadata", r.status_code == 200 and j["name"] == "ocr" and j["versions"] == ["1"])
    check("metadata io", j["inputs"][0]["name"] == "IMAGE"
          and [o["name"] for o in j["outputs"]] == ["OCR_RESULT", "TEXT"])
    check("versioned ready", c.get("/v2/models/ocr/versions/1/ready").status_code == 200)
    check("model config", c.get("/v2/models/ocr/config").status_code == 200)
    r = c.post("/v2/repository/index")
    check("repository index", r.status_code == 200 and r.json()[0]["state"] == "READY")
    r = c.get("/v2/models/nope")
    check("unknown model 404", r.status_code == 404 and "unknown model" in r.json()["error"])
    r = c.get("/v2/models/ocr/versions/7/ready")
    check("unknown version 404", r.status_code == 404, str(r.status_code))

    print("\nbinary inference")
    r = binary_infer(c, [PNG])
    check("binary 200", r.status_code == 200, r.text[:200])
    hl = int(r.headers["Inference-Header-Content-Length"])
    head = json.loads(r.content[:hl])
    body = r.content[hl:]
    check("response is binary", r.headers.get("content-type", "").startswith("application/octet-stream"))
    check("header names model", head["model_name"] == "ocr" and head["model_version"] == "1")
    check("two outputs", [o["name"] for o in head["outputs"]] == ["OCR_RESULT", "TEXT"])
    check("outputs are BYTES", all(o["datatype"] == "BYTES" for o in head["outputs"]))
    check("shape matches batch", all(o["shape"] == [1] for o in head["outputs"]))
    sizes = [o["parameters"]["binary_data_size"] for o in head["outputs"]]
    check("blob length matches sizes", sum(sizes) == len(body), f"{sum(sizes)} vs {len(body)}")
    res = decode_bytes(body[:sizes[0]])
    txt = decode_bytes(body[sizes[0]:])
    check("OCR_RESULT is json", json.loads(res[0])["results"][0]["text"] == "hello")
    check("TEXT is plain", txt[0] == b"hello", txt[0][:40])
    check("png content-type sniffed", seen.get("content_type") == "image/png", str(seen.get("content_type")))

    print("\njson inference")
    r = c.post("/v2/models/ocr/infer", json={
        "inputs": [{"name": "IMAGE", "datatype": "BYTES", "shape": [1],
                    "data": [base64.b64encode(PNG).decode()]}],
        "outputs": [{"name": "OCR_RESULT"}],
        "id": "req-42",
    })
    j = r.json()
    check("json 200", r.status_code == 200, r.text[:200])
    check("json stays json", r.headers["content-type"].startswith("application/json"))
    check("id echoed", j.get("id") == "req-42")
    check("single output", len(j["outputs"]) == 1 and j["outputs"][0]["name"] == "OCR_RESULT")
    # Triton's JSON BYTES form is an array of plain strings, not base64.
    check("json data is a string", isinstance(j["outputs"][0]["data"][0], str))
    decoded = json.loads(j["outputs"][0]["data"][0])
    check("json payload decodes", decoded["results"][0]["text"] == "hello")
    r = c.post("/v2/models/ocr/infer", json={
        "inputs": [{"name": "IMAGE", "datatype": "BYTES", "shape": [1],
                    "data": [base64.b64encode(PNG).decode()]}],
        "outputs": [{"name": "TEXT"}]})
    check("json TEXT is plain", r.json()["outputs"][0]["data"] == ["hello"], r.text[:120])

    print("\nparameters")
    r = binary_infer(c, [PNG], params={"layout": True, "tables": 1, "text": False})
    check("flags forwarded", seen["query"].get("layout") == "1" and seen["query"].get("tables") == "1"
          and seen["query"].get("text") == "0", str(seen["query"]))
    r = binary_infer(c, [PNG], params={"sequence_id": 7, "priority": 1})
    check("reserved params ignored", r.status_code == 200, r.text[:200])
    r = binary_infer(c, [PNG], params={"nonsense": 1})
    check("unknown param ignored by default", r.status_code == 200, r.text[:200])
    ta.STRICT_PARAMS = True
    r = binary_infer(c, [PNG], params={"nonsense": 1})
    check("strict rejects unknown param", r.status_code == 400 and "nonsense" in r.json()["error"])
    ta.STRICT_PARAMS = False
    r = binary_infer(c, [PNG], params={"layout": "banana"})
    check("bad bool rejected", r.status_code == 400, str(r.status_code))

    print("\nbatch")
    r = binary_infer(c, [PNG, PNG, PNG])
    hl = int(r.headers["Inference-Header-Content-Length"])
    head = json.loads(r.content[:hl])
    body = r.content[hl:]
    sizes = [o["parameters"]["binary_data_size"] for o in head["outputs"]]
    res = decode_bytes(body[:sizes[0]])
    check("batch used /ocr/batch", seen.get("batch_n") == 3, str(seen.get("batch_n")))
    check("3 elements out", len(res) == 3 and head["outputs"][0]["shape"] == [3])
    check("per-slot error kept in-band", json.loads(res[2]).get("error") == "decode_failed",
          res[2][:80].decode())
    check("good slots intact", json.loads(res[0])["results"][0]["text"] == "img0")

    print("\npdf")
    r = binary_infer(c, [PDF])
    hl = int(r.headers["Inference-Header-Content-Length"])
    head = json.loads(r.content[:hl])
    body = r.content[hl:]
    sizes = [o["parameters"]["binary_data_size"] for o in head["outputs"]]
    res = decode_bytes(body[:sizes[0]])
    txt = decode_bytes(body[sizes[0]:])
    check("pdf routed to /ocr/pdf", seen["path"] == "/ocr/pdf", seen["path"])
    check("pdf pages returned", len(json.loads(res[0])["pages"]) == 2)
    check("pdf pages formfeed-joined", txt[0] == b"p1\fp2", txt[0])

    print("\nerrors")
    r = c.post("/v2/models/ocr/infer", json={"inputs": [
        {"name": "IMAGE", "datatype": "FP32", "shape": [1], "data": [1.0]}]})
    check("wrong datatype 400", r.status_code == 400 and "BYTES" in r.json()["error"])
    r = c.post("/v2/models/ocr/infer", json={"inputs": [
        {"name": "WRONG", "datatype": "BYTES", "shape": [1], "data": ["aGk="]}]})
    check("missing input 400", r.status_code == 400 and "IMAGE" in r.json()["error"])
    r = c.post("/v2/models/ocr/infer", json={"inputs": [
        {"name": "IMAGE", "datatype": "BYTES", "shape": [2], "data": ["aGk="]}]})
    check("shape mismatch 400", r.status_code == 400 and "implies 2" in r.json()["error"])
    r = c.post("/v2/models/ocr/infer", json={"inputs": [
        {"name": "IMAGE", "datatype": "BYTES", "shape": [1], "data": ["aGk="]}],
        "outputs": [{"name": "SOMETHING"}]})
    check("unknown output 400", r.status_code == 400 and "SOMETHING" in r.json()["error"])
    blob = encode_bytes([PNG])
    head = json.dumps({"inputs": [{"name": "IMAGE", "datatype": "BYTES", "shape": [1],
                                  "parameters": {"binary_data_size": len(blob) + 999}}]}).encode()
    r = c.post("/v2/models/ocr/infer", content=head + blob,
               headers={"Inference-Header-Content-Length": str(len(head))})
    check("oversized binary_data_size 400", r.status_code == 400 and "runs past" in r.json()["error"])
    r = c.post("/v2/models/ocr/infer", content=b"{}",
               headers={"Inference-Header-Content-Length": "9999"})
    check("bogus header length 400", r.status_code == 400 and "exceeds" in r.json()["error"])

    print("\nbackend failures")

    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/ready":
            return httpx.Response(503)
        return httpx.Response(400, text="LAYOUT_DISABLED")

    ta.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    r = binary_infer(c, [PNG], params={"layout": True})
    check("backend 4xx passed through", r.status_code == 400 and "LAYOUT_DISABLED" in r.json()["error"],
          str(r.status_code))
    check("not-ready reported", c.get("/v2/health/ready").status_code == 400)
    check("live still 200 when backend down", c.get("/v2/health/live").status_code == 200)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL TRITON PROTOCOL CHECKS PASSED")
