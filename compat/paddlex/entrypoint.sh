#!/usr/bin/env bash
# Runs turboocr-server and the PaddleX-compatible adapter side by side.
#
# The adapter owns the externally advertised port (PADDLEX_PORT, default 8080 —
# PaddleX's own serving default) so existing clients need no change; TurboOCR
# itself binds a private port (TURBO_PORT, default 8081).
set -euo pipefail

TURBO_PORT="${TURBO_PORT:-8081}"
PADDLEX_PORT="${PADDLEX_PORT:-8080}"
export TURBO_OCR_URL="http://127.0.0.1:${TURBO_PORT}"

# Kill the whole process group on exit so a crash in either process takes the
# container down, rather than leaving a half-serving zombie that still passes
# a TCP health check.
_shutdown() {
    trap - TERM INT EXIT
    kill -TERM -$$ 2>/dev/null || true
}
trap _shutdown TERM INT EXIT

./build/turboocr-server --http-port "${TURBO_PORT}" &
TURBO_PID=$!

# Wait for readiness before accepting traffic. With a warm engine cache this is
# seconds; on a cold cache it is hours, so there is no timeout here — the
# container's own healthcheck decides when to give up.
echo "[entrypoint] waiting for turboocr-server on :${TURBO_PORT}"
while ! curl -fsS "http://127.0.0.1:${TURBO_PORT}/health/ready" >/dev/null 2>&1; do
    if ! kill -0 "$TURBO_PID" 2>/dev/null; then
        echo "[entrypoint] turboocr-server exited during startup" >&2
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] backend ready; starting PaddleX adapter on :${PADDLEX_PORT}"

exec uvicorn paddlex_adapter:app \
    --app-dir /app/compat/paddlex \
    --host 0.0.0.0 --port "${PADDLEX_PORT}" \
    --workers "${PADDLEX_WORKERS:-4}" \
    --log-level "${PADDLEX_LOG_LEVEL:-info}"
