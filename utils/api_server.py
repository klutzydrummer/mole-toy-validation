"""
Training metrics API server.

Exposes live training data from the checkpoints directory over HTTP so that
Claude Code can query live statistics without manual copy-paste.

Endpoints:
    GET /health              — liveness check, no auth required
    GET /report              — full structured report (report_agent.json)
    GET /metrics/latest      — last N log entries across all JSONL files (?n=50)
    GET /metrics/{config}    — all log entries for a specific config
    GET /status              — verify.py hash status + smoke test result

Auth: Bearer token via TRAINING_API_TOKEN env var.
      All endpoints except /health require the Authorization header.

Robustness:
    - Global exception handler: no unhandled exception ever crashes the server process.
    - Every route is wrapped in try/except: file I/O errors, JSON decode errors,
      and unexpected exceptions all return structured error responses, never 500 tracebacks
      that propagate to uvicorn's worker death path.
    - File reads are fully defensive: missing files, empty files, and partial writes
      (mid-rotation) are all handled gracefully.
    - NaN/Inf floats in training metrics are replaced with None before serialization
      so the JSON encoder never raises on non-finite values.

Usage (on the cloud instance):
    TRAINING_API_TOKEN=<token> uvicorn utils.api_server:app --host 0.0.0.0 --port 8787
"""

import glob
import json
import logging
import math
import os
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("api_server")

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
API_TOKEN = os.environ.get("TRAINING_API_TOKEN", "")

app = FastAPI(title="MoLE Training API", version="1.0")
_bearer = HTTPBearer(auto_error=False)


# ── Global exception handler ──────────────────────────────────────────────────
# Catches anything that escapes a route handler. Returns a structured JSON error
# instead of letting uvicorn propagate the exception to its worker management,
# which could cause worker death or restart loops.

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, tb)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ── Auth ──────────────────────────────────────────────────────────────────────

def require_token(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="TRAINING_API_TOKEN not configured on server")
    if creds is None or creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json_file(path: Path) -> Optional[dict | list]:
    """Read a JSON file defensively. Returns None on any error."""
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return json.loads(text)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        log.warning("JSON decode error reading %s: %s", path, e)
        return None
    except OSError as e:
        log.warning("OS error reading %s: %s", path, e)
        return None


def _load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts, skipping malformed lines.

    Handles: missing files, empty files, partial final lines from mid-write
    flushes, and individual corrupt records. Never raises.
    """
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.debug("Skipping malformed JSONL line %d in %s", lineno, path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("OS error reading JSONL %s: %s", path, e)
    return records


def _sanitize(obj):
    """Replace float NaN/Inf with None for safe JSON serialization.

    PyTorch training metrics can produce NaN grad norms. JSON does not support
    NaN/Inf as values, so they must be replaced before the response is encoded.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _find_jsonl_files() -> list[Path]:
    """Return all JSONL files in the checkpoint directory. Never raises."""
    try:
        return [Path(p) for p in glob.glob(str(CKPT_DIR / "*_seed*.jsonl"))]
    except OSError as e:
        log.warning("Error globbing JSONL files: %s", e)
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — no auth required. Always returns 200 if the process is alive."""
    try:
        ckpt_exists = CKPT_DIR.exists()
        jsonl_count = len(_find_jsonl_files())
        return {
            "status": "ok",
            "checkpoint_dir": str(CKPT_DIR),
            "checkpoint_dir_exists": ckpt_exists,
            "jsonl_files_found": jsonl_count,
        }
    except Exception as e:
        log.error("Unexpected error in /health: %s", e)
        # /health must always return 200 — even if something goes wrong internally
        return {"status": "ok", "warning": str(e)}


@app.get("/report")
def report(_=Depends(require_token)):
    """Return the full structured agent report (report_agent.json)."""
    try:
        path = CKPT_DIR / "report_agent.json"
        data = _read_json_file(path)
        if data is None:
            raise HTTPException(status_code=404, detail="report_agent.json not found or empty")
        return _sanitize(data)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error in /report: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to read report: {e}")


@app.get("/metrics/latest")
def metrics_latest(n: int = Query(50, ge=1, le=1000), _=Depends(require_token)):
    """Return the last n log entries across all JSONL files, sorted by step.

    Merges all config JSONL files, annotates each record with its config name,
    sorts by step, and returns the tail. Useful for a live training overview.
    """
    try:
        all_records = []
        for filepath in _find_jsonl_files():
            config = filepath.stem  # e.g. baseline_seed42
            for record in _load_jsonl(filepath):
                record["_config"] = config
                all_records.append(record)

        all_records.sort(key=lambda r: (r.get("step", 0), r.get("_config", "")))
        return _sanitize(all_records[-n:])
    except Exception as e:
        log.error("Unexpected error in /metrics/latest: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")


@app.get("/metrics/{config}")
def metrics_config(config: str, _=Depends(require_token)):
    """Return all log entries for a specific config.

    Accepts either the full run name ('baseline_seed42') or just the config
    prefix ('baseline') — will match all seeds if multiple exist.
    """
    try:
        # Sanitize config name to prevent path traversal
        safe = config.replace("/", "").replace("..", "").strip()
        if not safe:
            raise HTTPException(status_code=400, detail="Invalid config name")

        candidates = [Path(p) for p in glob.glob(str(CKPT_DIR / f"{safe}*.jsonl"))]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"No JSONL found for config '{safe}'")

        all_records = []
        for filepath in candidates:
            run_name = filepath.stem
            for record in _load_jsonl(filepath):
                record["_config"] = run_name
                all_records.append(record)

        all_records.sort(key=lambda r: (r.get("step", 0), r.get("_config", "")))
        return _sanitize(all_records)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error in /metrics/%s: %s", config, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")


@app.get("/status")
def status(_=Depends(require_token)):
    """Return verification hash status, smoke test result, and active config list."""
    try:
        result: dict = {}

        # Smoke test result
        result["smoke_test"] = _read_json_file(CKPT_DIR / "smoke_test_result.json")

        # Last verified record
        result["verification"] = _read_json_file(
            Path("references/verification/last_verified.json")
        )

        # Active JSONL files with their last recorded step
        active = {}
        for filepath in _find_jsonl_files():
            records = _load_jsonl(filepath)
            step_records = [r for r in records if "step" in r]
            last_step = step_records[-1]["step"] if step_records else None
            active[filepath.stem] = {"last_step": last_step, "record_count": len(records)}
        result["active_configs"] = active

        return _sanitize(result)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error in /status: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to read status: {e}")
