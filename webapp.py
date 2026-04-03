"""
Dataiku Webapp entry point (Flask).
Deploy this as a "Code webapp" in Dataiku DSS with backend type = Flask.

Routes:
  GET  /            → Chat UI
  POST /chat        → Send a text message
  POST /upload      → Upload a site list file (CSV/Excel)
  POST /confirm     → Explicit confirm/cancel/edit (used by confirm_dialog.js)
  POST /export      → Write result to a Dataiku dataset
  GET  /healthz     → Startup health check (returns init error if any)
"""
import logging
import os
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-prod")

# ---------------------------------------------------------------------------
# Lazy initialization — deferred until first request so that import-time
# errors surface as readable JSON rather than a silent 500.
# ---------------------------------------------------------------------------
_session_store = None
_orchestrator = None
_init_error = None


def _get_orchestrator():
    global _session_store, _orchestrator, _init_error
    if _orchestrator is not None:
        return _orchestrator, None
    if _init_error is not None:
        return None, _init_error
    try:
        from backend.orchestrator.orchestrator import Orchestrator
        from backend.state.session_store import SessionStore
        _session_store = SessionStore(timeout_minutes=30)
        _orchestrator = Orchestrator(_session_store)
        logger.info("Orchestrator initialized successfully.")
        return _orchestrator, None
    except Exception:
        _init_error = traceback.format_exc()
        logger.error("Orchestrator initialization failed:\n%s", _init_error)
        return None, _init_error


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    orch, err = _get_orchestrator()
    if err:
        return jsonify({"status": "error", "detail": err}), 500
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    orch, err = _get_orchestrator()
    if err:
        return jsonify({"error": f"Backend failed to initialize: {err}"}), 500

    data = request.get_json(force=True)
    session_id = data.get("session_id") or str(uuid.uuid4())
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    response = orch.process_message(session_id, user_message)
    return jsonify({"session_id": session_id, **response})


@app.route("/upload", methods=["POST"])
def upload():
    orch, err = _get_orchestrator()
    if err:
        return jsonify({"error": f"Backend failed to initialize: {err}"}), 500

    session_id = request.form.get("session_id") or str(uuid.uuid4())
    file_key = request.form.get("file_key", "")

    if file_key not in ("cro_file", "sponsor_file"):
        return jsonify({"error": "file_key must be 'cro_file' or 'sponsor_file'"}), 400

    if file_key not in request.files or request.files[file_key].filename == "":
        return jsonify({"error": "No file provided"}), 400

    response = orch.handle_file_upload(session_id, file_key, request.files[file_key])
    return jsonify({"session_id": session_id, **response})


@app.route("/confirm", methods=["POST"])
def confirm():
    orch, err = _get_orchestrator()
    if err:
        return jsonify({"error": f"Backend failed to initialize: {err}"}), 500

    data = request.get_json(force=True)
    session_id = data.get("session_id")
    confirmed = bool(data.get("confirmed", False))
    edit_params = data.get("edit_params")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    response = orch.handle_confirmation(session_id, confirmed, edit_params)
    return jsonify({"session_id": session_id, **response})


@app.route("/export", methods=["POST"])
def export():
    orch, err = _get_orchestrator()
    if err:
        return jsonify({"error": f"Backend failed to initialize: {err}"}), 500

    data = request.get_json(force=True)
    session_id = data.get("session_id")
    result_id = data.get("result_id")
    dataset_name = data.get("dataset_name", "").strip()

    if not session_id or not result_id or not dataset_name:
        return jsonify({"error": "Missing session_id, result_id, or dataset_name"}), 400

    response = orch.export_to_dataset(session_id, result_id, dataset_name)
    return jsonify({"session_id": session_id, **response})


# ---------------------------------------------------------------------------
# Dataiku webapp entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
