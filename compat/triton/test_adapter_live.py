"""End-to-end check of the Triton adapter against a live TurboOCR backend.

Exercises the wire protocol with plain httpx, then — if tritonclient is
installed — repeats the core path through the real client library, which is
the check that actually matters for a drop-in swap.

    ADAPTER_URL=http://localhost:8000 python3 compat/triton/test_adapter_live.py
"""
import base64, json, os, struct, sys, time
import httpx

ADAPTER = os.environ.get("ADAPTER_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("TRITON_MODEL_NAME", "ocr")
IMG = os.environ.get("TEST_IMAGE", "tests/fixtures/images/png/business_letter.png")
PDF = os.environ.get("TEST_PDF", "tests/fixtures/pdf/academic_paper.pdf")

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


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


def binary_infer(c, images, params=None, outputs=None):
    blob = encode_bytes(images)
    header = {"inputs": [{"name": "IMAGE", "datatype": "BYTES", "shape": [len(images)],
                          "parameters": {"binary_data_size": len(blob)}}]}
    if params:
        header["parameters"] = params
    if outputs is not None:
        header["outputs"] = outputs
    head = json.dumps(header).encode()
    r = c.post(f"{ADAPTER}/v2/models/{MODEL}/infer", content=head + blob,
               headers={"Inference-Header-Content-Length": str(len(head)),
                        "Content-Type": "application/octet-stream"})
    if r.status_code != 200 or "Inference-Header-Content-Length" not in r.headers:
        return r, None, None
    hl = int(r.headers["Inference-Header-Content-Length"])
    meta = json.loads(r.content[:hl])
    body = r.content[hl:]
    tensors, off = {}, 0
    for o in meta["outputs"]:
        n = o["parameters"]["binary_data_size"]
        tensors[o["name"]] = decode_bytes(body[off:off + n])
        off += n
    return r, meta, tensors


img_bytes = open(IMG, "rb").read()
pdf_bytes = open(PDF, "rb").read()

with httpx.Client(timeout=120) as c:
    print("metadata")
    r = c.get(f"{ADAPTER}/v2")
    check("GET /v2", r.status_code == 200 and "binary_tensor_data" in r.json().get("extensions", []),
          r.text[:150])
    check("live", c.get(f"{ADAPTER}/v2/health/live").status_code == 200)
    check("ready", c.get(f"{ADAPTER}/v2/health/ready").status_code == 200)
    check("model ready", c.get(f"{ADAPTER}/v2/models/{MODEL}/ready").status_code == 200)
    m = c.get(f"{ADAPTER}/v2/models/{MODEL}").json()
    check("model metadata", m.get("name") == MODEL and m.get("inputs", [{}])[0].get("name") == "IMAGE",
          json.dumps(m)[:200])
    check("repository index", c.post(f"{ADAPTER}/v2/repository/index").json()[0]["state"] == "READY")

    print("\nsingle image (binary)")
    t0 = time.perf_counter()
    r, meta, tensors = binary_infer(c, [img_bytes])
    dt = (time.perf_counter() - t0) * 1000
    check("infer 200", r.status_code == 200, r.text[:200])
    if tensors:
        doc = json.loads(tensors["OCR_RESULT"][0])
        text = tensors["TEXT"][0].decode()
        check("shape is [1]", all(o["shape"] == [1] for o in meta["outputs"]))
        check("results non-empty", len(doc.get("results", [])) > 0, str(len(doc.get("results", []))))
        check("TEXT matches results",
              text.split("\n")[0] == doc["results"][0]["text"] if doc.get("results") else False)
        print(f"    -> {len(doc.get('results', []))} lines, {dt:.0f} ms, first: {text.split(chr(10))[0]!r}")

    print("\nsingle image (json)")
    r = c.post(f"{ADAPTER}/v2/models/{MODEL}/infer", json={
        "id": "live-1",
        "inputs": [{"name": "IMAGE", "datatype": "BYTES", "shape": [1],
                    "data": [base64.b64encode(img_bytes).decode()]}],
        "outputs": [{"name": "TEXT"}]})
    j = r.json()
    check("json infer 200", r.status_code == 200, r.text[:200])
    check("json id echoed", j.get("id") == "live-1")
    check("json data is plain string", isinstance(j["outputs"][0]["data"][0], str)
          and not j["outputs"][0]["data"][0].startswith("eyJ"))

    print("\nparameters")
    r, meta, tensors = binary_infer(c, [img_bytes], params={"layout": True})
    if tensors:
        doc = json.loads(tensors["OCR_RESULT"][0])
        check("layout=1 emits layout", len(doc.get("layout", [])) > 0,
              "empty layout — is DISABLE_LAYOUT set?")
        check("layout_id on results", "layout_id" in (doc.get("results") or [{}])[0])

    print("\nbatch")
    r, meta, tensors = binary_infer(c, [img_bytes, img_bytes])
    check("batch 200", r.status_code == 200, r.text[:200])
    if tensors:
        check("2 elements out", len(tensors["OCR_RESULT"]) == 2
              and meta["outputs"][0]["shape"] == [2])
        check("both slots populated",
              all(len(json.loads(e).get("results", [])) > 0 for e in tensors["OCR_RESULT"]))

    print("\npdf")
    r, meta, tensors = binary_infer(c, [pdf_bytes])
    check("pdf 200", r.status_code == 200, r.text[:200])
    if tensors:
        doc = json.loads(tensors["OCR_RESULT"][0])
        check("pdf returns pages", len(doc.get("pages", [])) > 0, json.dumps(doc)[:150])
        check("pdf text formfeed-separated",
              "\f" in tensors["TEXT"][0].decode() or len(doc.get("pages", [])) == 1)

    print("\nerrors")
    r = c.get(f"{ADAPTER}/v2/models/not-a-model")
    check("unknown model 404", r.status_code == 404 and "error" in r.json(), str(r.status_code))
    r = c.post(f"{ADAPTER}/v2/models/{MODEL}/infer", json={"inputs": [
        {"name": "IMAGE", "datatype": "FP32", "shape": [1], "data": [1.0]}]})
    check("wrong datatype 400", r.status_code == 400 and "BYTES" in r.json()["error"], r.text[:150])
    r = c.post(f"{ADAPTER}/v2/models/{MODEL}/infer", json={"inputs": [
        {"name": "IMAGE", "datatype": "BYTES", "shape": [1], "data": ["not-an-image"]}]})
    check("undecodable image errors", r.status_code >= 400, str(r.status_code))

try:
    import numpy as np
    import tritonclient.http as httpclient
except ImportError:
    print("\ntritonclient not installed — skipping real-client checks")
else:
    print("\ntritonclient")
    host = ADAPTER.split("://", 1)[-1]
    cl = httpclient.InferenceServerClient(url=host)
    check("tc server live", cl.is_server_live())
    check("tc model ready", cl.is_model_ready(MODEL))
    check("tc metadata", cl.get_model_metadata(MODEL)["name"] == MODEL)
    arr = np.array([img_bytes], dtype=object)
    inp = httpclient.InferInput("IMAGE", [1], "BYTES")
    inp.set_data_from_numpy(arr, binary_data=True)
    res = cl.infer(MODEL, [inp], outputs=[httpclient.InferRequestedOutput("TEXT")])
    out = res.as_numpy("TEXT")
    check("tc infer returns text", out is not None and len(out[0]) > 0,
          repr(out[0][:60]) if out is not None else "None")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL LIVE TRITON CHECKS PASSED")
