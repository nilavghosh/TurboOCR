"""End-to-end check of the PaddleX adapter against a live TurboOCR backend."""
import base64, json, os, sys, time
import httpx

ADAPTER = os.environ.get("ADAPTER_URL", "http://127.0.0.1:9000")
IMG = os.environ.get("TEST_IMAGE", "tests/fixtures/images/png/receipt.png")
PDF = os.environ.get("TEST_PDF", "tests/fixtures/pdf/simple_letter.pdf")

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)

with httpx.Client(timeout=120) as c:
    # --- health ---
    r = c.get(f"{ADAPTER}/health")
    j = r.json()
    check("health 200", r.status_code == 200, str(r.status_code))
    check("health envelope", j.get("errorCode") == 0 and j.get("errorMsg") == "Healthy", json.dumps(j)[:200])
    check("health has logId", isinstance(j.get("logId"), str))

    # --- image ---
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    t0 = time.perf_counter()
    r = c.post(f"{ADAPTER}/ocr", json={"file": b64, "fileType": 1})
    dt = (time.perf_counter() - t0) * 1000
    j = r.json()
    check("image 200", r.status_code == 200, str(r.status_code) + " " + r.text[:200])
    check("envelope errorCode=0", j.get("errorCode") == 0)
    check("envelope errorMsg=Success", j.get("errorMsg") == "Success")
    res = j.get("result", {})
    check("has ocrResults", isinstance(res.get("ocrResults"), list) and len(res["ocrResults"]) == 1)
    check("dataInfo type=image", res.get("dataInfo", {}).get("type") == "image", json.dumps(res.get("dataInfo"))[:120])
    check("dataInfo dims", res.get("dataInfo", {}).get("width", 0) > 0 and res.get("dataInfo", {}).get("height", 0) > 0)

    entry = res["ocrResults"][0]
    check("image fields null", entry["ocrImage"] is None and entry["inputImage"] is None
          and entry["docPreprocessingImage"] is None)
    p = entry["prunedResult"]
    expected_keys = ["model_settings", "dt_polys", "text_det_params", "text_type",
                     "text_rec_score_thresh", "return_word_box", "rec_texts",
                     "rec_scores", "rec_polys", "rec_boxes"]
    check("prunedResult key order", list(p.keys()) == expected_keys, str(list(p.keys())))
    check("no input_path/page_index", "input_path" not in p and "page_index" not in p)
    check("rec_texts non-empty", len(p["rec_texts"]) > 0, str(len(p["rec_texts"])))
    check("parallel arrays", len(p["rec_texts"]) == len(p["rec_scores"]) == len(p["rec_polys"]) == len(p["rec_boxes"]))
    check("rec_boxes are [x1,y1,x2,y2]", all(len(b) == 4 and b[0] <= b[2] and b[1] <= b[3] for b in p["rec_boxes"]))
    check("rec_polys are 4-point", all(len(q) == 4 and all(len(pt) == 2 for pt in q) for q in p["rec_polys"]))
    check("text_type general", p["text_type"] == "general")
    check("orientation angles omitted", "textline_orientation_angles" not in p)
    check("no doc_preprocessor_res", "doc_preprocessor_res" not in p)
    print(f"    -> {len(p['rec_texts'])} lines, {dt:.0f} ms, first: {p['rec_texts'][0]!r}")

    # --- textRecScoreThresh actually filters ---
    r2 = c.post(f"{ADAPTER}/ocr", json={"file": b64, "fileType": 1, "textRecScoreThresh": 0.999})
    p2 = r2.json()["result"]["ocrResults"][0]["prunedResult"]
    check("threshold filters rec_*", len(p2["rec_texts"]) < len(p["rec_texts"]),
          f"{len(p2['rec_texts'])} vs {len(p['rec_texts'])}")
    check("threshold keeps dt_polys", len(p2["dt_polys"]) == len(p["dt_polys"]))
    check("threshold echoed", p2["text_rec_score_thresh"] == 0.999)

    # --- logId echoed ---
    r3 = c.post(f"{ADAPTER}/ocr", json={"file": b64, "fileType": 1, "logId": "my-trace-id"})
    check("logId echoed", r3.json().get("logId") == "my-trace-id")

    # --- PDF ---
    pb64 = base64.b64encode(open(PDF, "rb").read()).decode()
    r4 = c.post(f"{ADAPTER}/ocr", json={"file": pb64, "fileType": 0})
    j4 = r4.json()
    check("pdf 200", r4.status_code == 200, r4.text[:200])
    di = j4.get("result", {}).get("dataInfo", {})
    check("dataInfo type=pdf", di.get("type") == "pdf", json.dumps(di)[:150])
    check("numPages matches pages[]", di.get("numPages") == len(di.get("pages", [])))
    check("ocrResults per page", len(j4["result"]["ocrResults"]) == di.get("numPages"))

    # --- errors ---
    r5 = c.post(f"{ADAPTER}/ocr", json={"file": b64})
    check("422 undeterminable type", r5.status_code == 422 and r5.json()["errorCode"] == 422,
          str(r5.status_code))
    check("error has logId+errorMsg", "logId" in r5.json() and r5.json().get("errorMsg"))
    r6 = c.post(f"{ADAPTER}/ocr", json={"fileType": 1})
    check("422 missing file", r6.status_code == 422)
    r7 = c.post(f"{ADAPTER}/ocr", json={"file": "!!!not-base64!!!", "fileType": 1})
    check("422 bad base64/decode", r7.status_code in (422, 500), str(r7.status_code))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("ALL LIVE ADAPTER CHECKS PASSED")
