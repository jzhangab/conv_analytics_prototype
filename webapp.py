"""
Dataiku Webapp entry point (Flask).
Deploy this as a "Code webapp" in Dataiku DSS with backend type = Flask.

Routes:
  GET  /            → Chat UI
  POST /chat        → Send a text message
  POST /upload      → Upload a site list file (CSV/Excel)
  POST /confirm     → Explicit confirm/cancel/edit (used by confirm_dialog.js)
  POST /export      → Write result to a Dataiku dataset
"""
import logging
import os
import uuid

from flask import Flask, jsonify, render_template, request

from backend.orchestrator.orchestrator import Orchestrator
from backend.state.session_store import SessionStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-prod")

# Shared session store and orchestrator (initialized once per process)
_session_store = SessionStore(timeout_minutes=30)
_orchestrator = Orchestrator(_session_store)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    session_id = data.get("session_id") or str(uuid.uuid4())
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    response = _orchestrator.process_message(session_id, user_message)
    return jsonify({"session_id": session_id, **response})


@app.route("/upload", methods=["POST"])
def upload():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    file_key = request.form.get("file_key", "")  # "cro_file" or "sponsor_file"

    if file_key not in ("cro_file", "sponsor_file"):
        return jsonify({"error": "file_key must be 'cro_file' or 'sponsor_file'"}), 400

    if file_key not in request.files or request.files[file_key].filename == "":
        return jsonify({"error": "No file provided"}), 400

    response = _orchestrator.handle_file_upload(
        session_id, file_key, request.files[file_key]
    )
    return jsonify({"session_id": session_id, **response})


@app.route("/confirm", methods=["POST"])
def confirm():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    confirmed = bool(data.get("confirmed", False))
    edit_params = data.get("edit_params")  # optional dict

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    response = _orchestrator.handle_confirmation(session_id, confirmed, edit_params)
    return jsonify({"session_id": session_id, **response})


@app.route("/export", methods=["POST"])
def export():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    result_id = data.get("result_id")
    dataset_name = data.get("dataset_name", "").strip()

    if not session_id or not result_id or not dataset_name:
        return jsonify({"error": "Missing session_id, result_id, or dataset_name"}), 400

    response = _orchestrator.export_to_dataset(session_id, result_id, dataset_name)
    return jsonify({"session_id": session_id, **response})


# ---------------------------------------------------------------------------
# Dataiku webapp entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
