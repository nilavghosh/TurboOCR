"""Locust load test for TurboOCR.

Users are set with Locust's own flags, so nothing here needs editing to scale:

    locust -f locustfile.py --headless -u 25 -r 5 -t 2m --host http://localhost:8080

  -u / --users        concurrent users
  -r / --spawn-rate   users started per second
  -t / --run-time     duration (e.g. 90s, 2m, 1h)

Extra TurboOCR-specific flags (all optional):

  --ocr-endpoint /ocr/raw     endpoint under test
  --ocr-image <path>          single image to send
  --ocr-image-dir <dir>       directory of images, picked round-robin
  --ocr-warmup 0              per-user warmup requests, excluded from stats

A response only counts as a success when it is HTTP 200 *and* the JSON body
carries a `results` array — a 200 with an error payload would otherwise be
scored as a healthy request and quietly inflate the numbers.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from locust import HttpUser, constant, events, task

DEFAULT_IMAGE_DIR = "/workspace/TurboOCR/tests/fixtures/images/png"
DEFAULT_IMAGE = f"{DEFAULT_IMAGE_DIR}/business_letter.png"

_CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--ocr-endpoint", default="/ocr/raw",
                        help="Endpoint to load (default: /ocr/raw)")
    parser.add_argument("--ocr-image", default=None,
                        help=f"Single image to POST (default: {DEFAULT_IMAGE})")
    parser.add_argument("--ocr-image-dir", default=None,
                        help="Directory of images, cycled round-robin instead of one file")
    parser.add_argument("--ocr-warmup", type=int, default=0,
                        help="Warmup requests per user, excluded from reported stats")


# Payloads are read once at startup: re-reading per request would benchmark the
# local disk instead of the server.
_PAYLOADS: list[tuple[bytes, str]] = []
_CYCLE = None


@events.init.add_listener
def _(environment, **_kwargs):
    global _CYCLE
    opts = environment.parsed_options
    if opts.ocr_image_dir:
        paths = sorted(
            p for p in Path(opts.ocr_image_dir).iterdir()
            if p.suffix.lower() in _CONTENT_TYPES
        )
        if not paths:
            raise ValueError(f"no images found in {opts.ocr_image_dir}")
    else:
        paths = [Path(opts.ocr_image or DEFAULT_IMAGE)]

    for p in paths:
        _PAYLOADS.append((p.read_bytes(), _CONTENT_TYPES[p.suffix.lower()]))
    _CYCLE = itertools.cycle(range(len(_PAYLOADS)))

    total_mb = sum(len(b) for b, _ in _PAYLOADS) / 1e6
    print(f"[locust] endpoint={opts.ocr_endpoint} images={len(_PAYLOADS)} "
          f"({total_mb:.1f} MB in memory) warmup={opts.ocr_warmup}/user")


class OCRUser(HttpUser):
    # No think time: this measures server saturation throughput, not a
    # simulated human browsing pattern.
    wait_time = constant(0)

    def on_start(self):
        opts = self.environment.parsed_options
        self.endpoint = opts.ocr_endpoint
        # Stagger user start so all N users do not fire their first request on
        # the same tick, which would show up as a false latency spike.
        if len(_PAYLOADS) > 1:
            self.offset = random.randrange(len(_PAYLOADS))
        else:
            self.offset = 0
        for _ in range(opts.ocr_warmup):
            body, ct = _PAYLOADS[next(_CYCLE)]
            self.client.post(self.endpoint, data=body,
                             headers={"Content-Type": ct},
                             name="warmup (excluded)")

    @task
    def ocr(self):
        body, ct = _PAYLOADS[next(_CYCLE)]
        with self.client.post(self.endpoint, data=body,
                              headers={"Content-Type": ct},
                              name=self.endpoint,
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            # A 200 is not proof of a real OCR result; verify the payload shape.
            try:
                payload = json.loads(r.text)
            except json.JSONDecodeError:
                r.failure("response was not JSON")
                return
            if not isinstance(payload.get("results"), list):
                r.failure("JSON had no 'results' array")
                return
            r.success()
