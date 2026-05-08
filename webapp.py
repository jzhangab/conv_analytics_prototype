"""
Flask HTTP adapter.

This file is intentionally thin.  Its only job is to translate HTTP requests
into ChatRequest objects, call backend.process(), and return the ChatResponse
as JSON.  All business logic lives in backend/api/chat_backend.py.

Routes
------
  POST /api/interact   — unified endpoint for the React frontend
  GET  /healthz        — startup health check
  GET  /               — React SPA stub (serve index.html or redirect)

Legacy routes (kept for the existing Dataiku frontend during migration):
  POST /chat    →  action="message"
  POST /upload  →  action="upload"
  POST /confirm →  action="confirm"
  POST /export  →  action="export"
"""
import logging
import os
import uuid

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.api.chat_backend import ChatBackend
from backend.api.models import ChatRequest, UploadedFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="frontend/static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-prod")
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","))

# ---------------------------------------------------------------------------
# Single ChatBackend instance — initialized lazily on first request so that
# startup errors surface as readable JSON rather than a silent crash.
# ---------------------------------------------------------------------------
_backend: ChatBackend | None = None
_init_error: str | None = None


def _get_backend() -> tuple[ChatBackend | None, str | None]:
    global _backend, _init_error
    if _backend is not None:
        return _backend, None
    if _init_error is not None:
        return None, _init_error
    try:
        _backend = ChatBackend()
        return _backend, None
    except Exception:
        import traceback
        _init_error = traceback.format_exc()
        logger.error("ChatBackend init failed:\n%s", _init_error)
        return None, _init_error


def _guard() -> tuple[ChatBackend | None, dict | None]:
    """Return (backend, None) on success or (None, error_response) on failure."""
    backend, err = _get_backend()
    if err:
        return None, ({"success": False, "error": f"Backend init failed: {err}"}, 500)
    return backend, None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    _, err = _get_backend()
    if err:
        return jsonify({"status": "error", "detail": err}), 500
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# SPA stub — remove or replace with your React build's index.html
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    # If React is built into frontend/static, serve it.  Otherwise 204.
    static_index = os.path.join(app.static_folder, "index.html")
    if os.path.exists(static_index):
        return send_from_directory(app.static_folder, "index.html")
    return ("", 204)


# ---------------------------------------------------------------------------
# Unified endpoint (React frontend)
# ---------------------------------------------------------------------------

@app.route("/api/interact", methods=["POST"])
def interact():
    """
    Single endpoint for all React frontend interactions.

    JSON body for action="message" | "confirm" | "export":
        { "session_id": "...", "action": "...", ...action-specific fields... }

    Multipart form-data for action="upload":
        session_id=..., file_key=..., <file_key>=<file bytes>
    """
    backend, err_resp = _guard()
    if err_resp:
        return jsonify(err_resp[0]), err_resp[1]

    content_type = request.content_type or ""
    if "multipart/form-data" in content_type:
        req, bad = _parse_upload_request()
    else:
        req, bad = _parse_json_request()

    if bad:
        return jsonify(bad[0]), bad[1]

    return jsonify(backend.process(req).to_dict())


# ---------------------------------------------------------------------------
# Legacy routes (backward-compatible with the existing Dataiku frontend)
# Remove these once the React frontend is fully wired up.
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat_legacy():
    backend, err_resp = _guard()
    if err_resp:
        return jsonify(err_resp[0]), err_resp[1]
    data = request.get_json(force=True) or {}
    req = ChatRequest(
        session_id=data.get("session_id") or str(uuid.uuid4()),
        action="message",
        message=data.get("message", ""),
    )
    return jsonify(backend.process(req).to_dict())


@app.route("/upload", methods=["POST"])
def upload_legacy():
    backend, err_resp = _guard()
    if err_resp:
        return jsonify(err_resp[0]), err_resp[1]
    req, bad = _parse_upload_request()
    if bad:
        return jsonify(bad[0]), bad[1]
    return jsonify(backend.process(req).to_dict())


@app.route("/confirm", methods=["POST"])
def confirm_legacy():
    backend, err_resp = _guard()
    if err_resp:
        return jsonify(err_resp[0]), err_resp[1]
    data = request.get_json(force=True) or {}
    req = ChatRequest(
        session_id=data.get("session_id") or str(uuid.uuid4()),
        action="confirm",
        confirmed=bool(data.get("confirmed", False)),
        edit_params=data.get("edit_params"),
    )
    return jsonify(backend.process(req).to_dict())


@app.route("/export", methods=["POST"])
def export_legacy():
    backend, err_resp = _guard()
    if err_resp:
        return jsonify(err_resp[0]), err_resp[1]
    data = request.get_json(force=True) or {}
    req = ChatRequest(
        session_id=data.get("session_id") or str(uuid.uuid4()),
        action="export",
        result_id=data.get("result_id"),
        export_destination=data.get("dataset_name") or data.get("table_name"),
    )
    return jsonify(backend.process(req).to_dict())


# ---------------------------------------------------------------------------
# Request parsers
# ---------------------------------------------------------------------------

def _parse_json_request() -> tuple[ChatRequest | None, tuple | None]:
    data = request.get_json(force=True) or {}
    action = data.get("action", "message")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if action == "message":
        if not data.get("message"):
            return None, ({"success": False, "error": "message field required"}, 400)
        return ChatRequest(
            session_id=session_id, action="message", message=data["message"]
        ), None

    if action == "confirm":
        if data.get("confirmed") is None:
            return None, ({"success": False, "error": "confirmed field required"}, 400)
        return ChatRequest(
            session_id=session_id, action="confirm",
            confirmed=bool(data["confirmed"]), edit_params=data.get("edit_params"),
        ), None

    if action == "export":
        return ChatRequest(
            session_id=session_id, action="export",
            result_id=data.get("result_id"),
            export_destination=data.get("export_destination") or data.get("table_name"),
        ), None

    return None, ({"success": False, "error": f"Unknown action: {action!r}"}, 400)


def _parse_upload_request() -> tuple[ChatRequest | None, tuple | None]:
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    file_key = request.form.get("file_key", "")

    if file_key not in ("site_file", "protocol_file"):
        return None, (
            {"success": False, "error": "file_key must be 'site_file' or 'protocol_file'"}, 400
        )
    if file_key not in request.files or not request.files[file_key].filename:
        return None, ({"success": False, "error": "No file provided"}, 400)

    fs = request.files[file_key]
    uploaded = UploadedFile(
        file_key=file_key,
        filename=fs.filename,
        data=fs.read(),
        content_type=fs.content_type or "application/octet-stream",
    )
    return ChatRequest(session_id=session_id, action="upload", files=[uploaded]), None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
