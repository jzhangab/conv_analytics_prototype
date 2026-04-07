"""
Generate a PDF flow diagram focused on the Notebook + Backend workflow.
No Flask webapp routes — only Panel notebook cells and the backend pipeline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Palette ──────────────────────────────────────────────────────────────────
C_NOTEBOOK  = "#2C3E7A"   # dark blue  — notebook cells
C_PANEL     = "#4A90D9"   # mid blue   — Panel UI widgets
C_ORCH      = "#7055C8"   # purple     — orchestrator / FSM
C_LLM       = "#C8861A"   # amber      — LLM client / prompts
C_AGENT     = "#2E8A52"   # green      — skill agents
C_DATA      = "#B04040"   # red        — data / config
C_OUTPUT    = "#5A7080"   # slate      — outputs
C_ARROW     = "#333333"
BG          = "#F6F7FB"
WHITE       = "#FFFFFF"

fig, ax = plt.subplots(figsize=(18, 32))
ax.set_xlim(0, 18)
ax.set_ylim(0, 32)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, title, sub=None, fill=WHITE, edge="#333",
        fs=8.5, bold=False, r=0.22, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.04,rounding_size={r}",
                          facecolor=fill, edgecolor=edge, linewidth=1.5,
                          zorder=3, alpha=alpha)
    ax.add_patch(rect)
    cy = y + h / 2
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(x + w/2, cy + h*0.14, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, zorder=4)
        ax.text(x + w/2, cy - h*0.18, sub, ha="center", va="center",
                fontsize=fs - 1.5, color="#555", zorder=4, style="italic",
                linespacing=1.3)
    else:
        ax.text(x + w/2, cy, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, zorder=4, linespacing=1.3)


def hdr(x, y, w, h, label, color, fs=10):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.04,rounding_size=0.18",
                          facecolor=color, edgecolor=color, linewidth=0,
                          alpha=0.88, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=WHITE, zorder=3)


def band(x, y, w, h, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="square,pad=0",
                          facecolor=color, edgecolor="none", zorder=1, alpha=0.12)
    ax.add_patch(rect)


def arr(x1, y1, x2, y2, lbl="", color=C_ARROW, lw=1.5, rad=0.0, dashed=False):
    style = f"arc3,rad={rad}"
    ls = (0, (4, 3)) if dashed else None
    props = dict(arrowstyle="-|>", color=color, lw=lw, connectionstyle=style)
    if dashed:
        props["linestyle"] = ls
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=5)
    if lbl:
        mx = (x1 + x2) / 2 + 0.08
        my = (y1 + y2) / 2
        ax.text(mx, my, lbl, fontsize=7, color="#333", zorder=6,
                ha="left", va="center",
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.9, pad=1))


def note(x, y, txt, color="#555", fs=7.5, align="left"):
    ax.text(x, y, txt, fontsize=fs, color=color, ha=align, va="center",
            zorder=6, style="italic")


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(9, 31.55, "Clinical Analytics Chatbot — Notebook & Backend Flow",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1a1a2e")
ax.text(9, 31.15,
        "conv_analytics_chatbot.ipynb  ·  Panel UI  ·  Orchestrator  ·  LLM Mesh  ·  Skill Agents  ·  Protocol Analysis  ·  Data Reasoning",
        ha="center", va="center", fontsize=9, color="#555")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — NOTEBOOK STARTUP (cells 1-3)  y: 28.5–30.9
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 30.3, 17.4, 0.50, "NOTEBOOK STARTUP  (run cells top-to-bottom once)", C_NOTEBOOK, fs=10)
band(0.3, 28.5, 17.4, 1.75, C_NOTEBOOK)

box(0.5, 28.55, 3.5, 1.55,
    "Cell 1 — Panel Init",
    "pn.extension('bokeh')\nraw_css → chat width fix\nsizing_mode='stretch_width'",
    fill="#E8EDF8", edge=C_NOTEBOOK, bold=True, fs=8)

box(4.3, 28.55, 3.5, 1.55,
    "Cell 2 — Config",
    "LLM_CONNECTION_ID\n= 'YOUR_LLM_MESH_ID'\n(user sets this)",
    fill="#E8EDF8", edge=C_NOTEBOOK, bold=True, fs=8)

box(8.1, 28.55, 9.4, 1.55,
    "Cell 3 — Backend Init + LLM Patch",
    "load llm_config.yaml · inject connection_id · flush cached backend modules\n"
    "SessionStore(60 min) · Orchestrator(session_store, config)\n"
    "monkey-patch LLMClient.complete → appends {messages, response} to call_log[]",
    fill="#E8EDF8", edge=C_NOTEBOOK, bold=True, fs=8)

arr(4.0, 29.32, 4.3, 29.32, color=C_NOTEBOOK, lw=1.8)
arr(7.8, 29.32, 8.1, 29.32, color=C_NOTEBOOK, lw=1.8)
note(0.45, 28.38, "Run once per session — flushes sys.modules so latest backend code is always loaded", color=C_NOTEBOOK)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — PANEL UI  y: 24.3–28.0
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 27.85, 17.4, 0.45, "CELL 4 — PANEL UI  (interactive runtime)", C_PANEL, fs=10)
band(0.3, 24.3, 17.4, 3.5, C_PANEL)

box(0.5, 26.5, 17.0, 0.9,
    "App Layout  ·  pn.Column",
    "Header · CRO Upload bar · Protocol Upload bar · ChatInterface · "
    "Export bar · Protocol PDF row · LLM Trace pane",
    fill="#D6E8FA", edge=C_PANEL, fs=8.5, bold=True)

# Six UI component boxes
box(0.5, 24.4, 2.6, 1.9,
    "ChatInterface",
    "height=720\ncallback=chat_callback\n+periodic_callback(2s)\nfor live trace",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

box(3.4, 24.4, 2.8, 1.9,
    "CRO Upload Bar",
    "FileInput(.csv/.xlsx)\n_on_cro_upload()\nparse_uploaded_file()\n→ sets site_file",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

box(6.5, 24.4, 2.8, 1.9,
    "Protocol Upload Bar",
    "FileInput(.pdf/.docx/.txt)\n_on_protocol_upload()\n→ handle_file_upload\n('protocol_file')",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

box(9.6, 24.4, 2.6, 1.9,
    "Confirm Buttons",
    "✓ Yes / ✗ Cancel\ncreated inside\n_response_to_panel()\nwhen fsm=CONFIRM",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

box(12.5, 24.4, 2.5, 1.9,
    "Export Bar +\nPDF Download",
    "Export → Dataiku dataset\nPDF row appears after\nprotocol_analysis\n→ FileDownload btn",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

box(15.3, 24.4, 2.2, 1.9,
    "LLM Trace\nPane",
    "pn.pane.Markdown\nmonospace scroll\nlive via periodic\ncallback (2 s)",
    fill="#D6E8FA", edge=C_PANEL, fs=7.2)

for cx in [1.8, 4.8, 7.9, 10.9, 13.75, 16.4]:
    arr(cx, 26.5, cx, 26.3, color=C_PANEL, lw=1.2)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — REQUEST FLOW  y: 21.0–24.0
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 23.85, 17.4, 0.45, "REQUEST FLOW  (chat_callback → orchestrator)", C_ORCH, fs=10)
band(0.3, 21.0, 17.4, 2.8, C_ORCH)

box(0.5, 22.6, 3.5, 1.1,
    "chat_callback(contents)",
    "periodic_callback active\nprocess_message()\n→ stop callback",
    fill="#EDE8FF", edge=C_ORCH, fs=8)

box(0.5, 21.1, 3.5, 1.2,
    "_on_cro_upload / _on_protocol_upload",
    "parse file → session state\n→ handle_file_upload(key, file)",
    fill="#EDE8FF", edge=C_ORCH, fs=7.5)

box(4.7, 21.1, 5.0, 2.7,
    "Orchestrator\norchestrator.py",
    None,
    fill="#EDE8FF", edge=C_ORCH, bold=True, fs=9.5)

ax.text(7.2, 23.4,  "process_message(session_id, text)", fontsize=7.2, ha="center", color="#333", zorder=4)
ax.text(7.2, 23.05, "handle_file_upload(session_id, key, file)", fontsize=7.2, ha="center", color="#333", zorder=4)
ax.text(7.2, 22.7,  "handle_confirmation(session_id, confirmed)", fontsize=7.2, ha="center", color="#333", zorder=4)
ax.text(7.2, 22.35, "export_to_dataset(session_id, result_id, name)", fontsize=7.2, ha="center", color="#333", zorder=4)
ax.plot([4.85, 9.55], [22.1, 22.1], color=C_ORCH, lw=0.7, alpha=0.4, zorder=4)
ax.text(7.2, 21.7,  "_route_fsm() → data_reasoning bypass\nor _check_and_confirm() → _execute_skill()", fontsize=7.2, ha="center", color="#333", zorder=4, linespacing=1.4)
ax.text(7.2, 21.2,  "skill_id in response · _build_response()", fontsize=7.2, ha="center", color="#555", zorder=4, style="italic")

arr(4.0, 23.15, 4.7, 23.15, "process_message", color=C_ORCH, lw=1.3)
arr(4.0, 21.7,  4.7, 22.0,  "handle_file_upload", color=C_ORCH, lw=1.3)

box(10.3, 22.4, 4.0, 1.4,
    "SessionStore  +  ConversationState",
    "session_id → state\nhistory · params · uploaded_files\nfsm_state · prior_results[]",
    fill="#EDE8FF", edge=C_ORCH, fs=7.5)

box(10.3, 21.1, 4.0, 1.0,
    "SkillSchema  (skills_config.yaml)",
    "required / optional param specs\nget_missing_required(params)",
    fill="#EDE8FF", edge=C_ORCH, fs=7.5)

arr(9.7, 22.8, 10.3, 22.8, color=C_ORCH)
arr(9.7, 21.6, 10.3, 21.6, color=C_ORCH)

arr(9.9, 24.4, 9.8, 23.7, color=C_ORCH, lw=1.1)
arr(9.8, 23.5, 9.8, 23.35, "handle_confirmation", color=C_ORCH, lw=1.1)

arr(13.75, 24.4, 11.0, 22.7, "export_to_dataset", color=C_ORCH, lw=1.0, rad=-0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — FSM  y: 17.6–20.7
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 20.65, 17.4, 0.42, "FINITE STATE MACHINE  (FSM)  +  Data Reasoning bypass", C_ORCH, fs=9.5)
band(0.3, 17.6, 17.4, 3.0, C_ORCH)

STATE_W, STATE_H = 2.8, 1.05
state_xs = [0.5, 3.8, 7.1, 10.5]
state_labels = ["IDLE", "PARAMETER\nGATHERING", "CONFIRMATION\nPENDING", "SKILL\nEXECUTION"]
for sx, sl in zip(state_xs, state_labels):
    box(sx, 19.35, STATE_W, STATE_H, sl, fill="#EDE8FF", edge=C_ORCH, fs=9, bold=True)

arr(3.3, 19.87, 3.8, 19.87, "intent\nrecognized", color=C_ORCH, lw=1.3)
arr(6.6, 19.87, 7.1, 19.87, "all params\npresent", color=C_ORCH, lw=1.3)
arr(9.9, 19.87, 10.5, 19.87, "user\nconfirms", color=C_ORCH, lw=1.3)

ax.annotate("", xy=(1.9, 19.35), xytext=(11.9, 19.35),
            arrowprops=dict(arrowstyle="-|>", color=C_ORCH, lw=1.2,
                            connectionstyle="arc3,rad=0.4"), zorder=5)
note(6.5, 18.6, "→ IDLE after execution / cancel", color=C_ORCH)

fn_labels = [
    "classify_intent()\n_parse_skill_selection()",
    "extract_parameters()\nask_for_missing()",
    "build_confirmation_prompt()\nparse_confirmation_reply()",
    "router.get_agent()\nagent.run(params, state)",
]
for sx, fl in zip(state_xs, fn_labels):
    box(sx, 17.65, STATE_W, 1.55, fl, fill="#F4F0FF", edge=C_ORCH, fs=7.2)
    arr(sx + STATE_W/2, 19.35, sx + STATE_W/2, 19.2, color=C_ORCH, lw=1.0)

# Data Reasoning bypass box
box(13.6, 18.4, 4.0, 1.95,
    "Data Reasoning\nBypass",
    "intent='data_reasoning'\nskips FSM param gathering\n_handle_reasoning()\nformats prior_results[]\nas context → single LLM call",
    fill="#EDE8FF", edge=C_ORCH, fs=7.5)
arr(7.1, 21.1, 15.6, 20.35, "data_reasoning\nintent", color=C_ORCH, lw=1.2, rad=-0.15, dashed=True)

arr(7.1, 21.1, 5.9, 20.65, "calls _route_fsm()", color=C_ORCH, lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — LLM CLIENT  y: 14.3–17.3
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 17.25, 17.4, 0.42, "LLM CLIENT  (llm_client.py)  +  Prompt Templates", C_LLM, fs=9.5)
band(0.3, 14.3, 17.4, 2.9, C_LLM)

box(0.5, 15.6, 4.0, 1.5,
    "LLMClient.complete(messages)",
    "→ dataiku.api_client()\n→ project.get_llm(connection_id)\n→ llm.new_completion()\n→ completion.execute()",
    fill="#FFF3D6", edge=C_LLM, fs=7.5, bold=True)

box(4.9, 15.6, 3.8, 1.5,
    "complete_json()\n_parse_json(raw)",
    "strip ``` fences\njson.loads(text)\n→ _repair_json() fallback\nif truncated",
    fill="#FFF3D6", edge=C_LLM, fs=7.5)

box(9.0, 15.6, 3.0, 1.5,
    "_repair_json(text)",
    "walk chars tracking\nopen { [ and strings\nclose unclosed structures\n(token cutoff recovery)",
    fill="#FFF3D6", edge=C_LLM, fs=7.5)

box(12.4, 15.6, 5.1, 1.5,
    "call_log  [ ]",
    "real calls: monkey-patched complete()\nappends {messages, response, label?}\nsynthetic: _trace() from agents\nperiodic_callback reads → Trace pane",
    fill="#FFF3D6", edge=C_LLM, fs=7.5)

arr(4.5, 16.35, 4.9, 16.35, color=C_LLM, lw=1.3)
arr(8.7, 16.35, 9.0, 16.35, color=C_LLM, lw=1.3)
arr(12.0, 16.35, 12.4, 16.35, color=C_LLM, lw=1.3)

box(0.5, 14.35, 8.3, 1.0,
    "completion.with_max_output_tokens(max_tokens)  ·  temp_classify / temp_extract / temp_agents / temp_deterministic from llm_config.yaml",
    fill="#FFE8A0", edge=C_LLM, fs=7.8)

arr(2.5, 15.6, 2.5, 15.35, color=C_LLM, lw=1.0)

box(9.1, 14.35, 8.4, 1.0,
    "prompt_templates.py  (10 template groups)",
    "INTENT_CLASSIFIER · PARAMETER_EXTRACTOR · CLARIFICATION_MESSAGE\n"
    "SITE_MATCHING · TRIAL_BENCHMARKING · DRUG_REIMBURSEMENT · ENROLLMENT_PARAMS/NARRATIVE\n"
    "DATA_REASONING · PROTOCOL_ANALYSIS · PROTOCOL_CHUNK (extraction, legacy)",
    fill="#FFF3D6", edge=C_LLM, fs=7.2)

arr(12.0, 14.85, 11.5, 14.85, color=C_LLM, lw=0.9)

arr(2.0, 17.65, 2.0, 17.1, "classify_intent()\nextract_params()", color=C_LLM, lw=1.3)
arr(9.0, 17.65, 2.5, 17.15, color=C_LLM, lw=1.0, rad=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — SKILL AGENTS  y: 9.8–14.0
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 13.92, 17.4, 0.42, "SKILL AGENTS  (router.py → agents/)  ·  BaseAgent.run(params, state)", C_AGENT, fs=9.5)
band(0.3, 9.8, 17.4, 4.1, C_AGENT)

box(0.5, 12.0, 2.5, 1.5,
    "Router",
    "skill_id\n→ agent instance",
    fill="#D8F0E0", edge=C_AGENT, bold=True, fs=8)

AGENT_DATA = [
    ("Site List\nMatching\nAgent",
     "LLM semantic match\nuploaded rows ↔ CTMS\nparse_site_matching_response()\n→ match table"),
    ("Trial\nBenchmarking\nAgent",
     "Citeline DB query\nLLM interprets metrics\n→ key_metrics JSON\n→ summary text"),
    ("Drug\nReimbursement\nAgent",
     "LLM HTA assessment\nper-country outlook\nlikelihood · risks\ntimeline months"),
    ("Enrollment\nForecasting\nAgent",
     "Stage 1: LLM estimates\nenroll rate / ramp / dropout\nStage 2: math model\n→ Bokeh chart + narrative"),
    ("Protocol\nAnalysis\nAgent",
     "Full text → 3M char cap\nSingle PROTOCOL_ANALYSIS\nLLM call · findings JSON\n→ text + table + PDF"),
]

AX = [3.3, 6.1, 8.9, 11.7, 14.5]
AW = 2.6
for i, (name, sub) in enumerate(AGENT_DATA):
    box(AX[i], 10.65, AW, 2.7, name, sub, fill="#D8F0E0", edge=C_AGENT, fs=7.3)
    arr(1.75, 12.5, AX[i] + AW/2, 13.35, color=C_AGENT, lw=1.0)

# Data sources row
box(3.3, 9.85, AW, 0.65, "CTMS_SITES.csv\n(300 sponsor sites)", fill="#FFE0D6", edge=C_DATA, fs=7.2)
box(6.1, 9.85, AW, 0.65, "Citeline DB\n(mock query results)", fill="#FFE0D6", edge=C_DATA, fs=7.2)
box(14.5, 9.85, AW, 0.65, "Enrollment math model\n(logistic ramp)", fill="#FFE0D6", edge=C_DATA, fs=7.2)

for dx in [3.3, 6.1, 14.5]:
    arr(dx + AW/2, 10.65, dx + AW/2, 10.5, color=C_DATA, lw=1.0)

box(0.5, 9.85, 2.5, 0.65, "response_parser.py", fill="#FFE0D6", edge=C_DATA, fs=7.5)
arr(1.75, 10.65, 1.75, 10.5, color=C_DATA, lw=1.0)

# Protocol file note
box(11.7, 9.85, AW, 0.65, "parse_protocol_file()\n(PDF/DOCX/TXT → pages[])", fill="#FFE0D6", edge=C_DATA, fs=7.2)
arr(15.8, 10.65, 15.8, 10.5, color=C_DATA, lw=1.0)

arr(11.9, 17.65, 1.75, 13.5, "router.get_agent(skill_id)", color=C_AGENT, lw=1.3, rad=0.2)
arr(8.5, 13.35, 4.0, 15.6, "LLM calls\nfrom agents", color=C_LLM, lw=1.0, rad=-0.1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G — OUTPUTS  y: 6.4–9.5
# ═══════════════════════════════════════════════════════════════════════════════
hdr(0.3, 9.55, 17.4, 0.42, "OUTPUTS  →  AgentResult  →  SkillResult  →  _response_to_panel()", C_OUTPUT, fs=9.5)
band(0.3, 6.4, 17.4, 3.1, C_OUTPUT)

OUTPUT_DATA = [
    ("Site Match Table",
     "Row · Uploaded Site\nMatch Status · CTMS Site ID\nConfidence · Basis\npn.pane.DataFrame"),
    ("Benchmark\nJSON → text",
     "key_metrics (rates, duration)\nnotable_patterns\nkey_challenges\npn.pane.Markdown"),
    ("Reimbursement\nJSON → text",
     "country_assessments\nlikelihood · requirements\nrisks · timeline\npn.pane.Markdown"),
    ("Enrollment\nForecast",
     "Bokeh figure (3 curves)\npessimistic/moderate/optimistic\n+ narrative paragraphs\npn.pane.Bokeh + Markdown"),
    ("Protocol\nAnalysis Report",
     "overall_rating · executive_summary\nstrengths · critical_concerns\nsection_assessments\n+ findings table + PDF download"),
]

OX = [0.5, 3.8, 7.1, 10.4, 13.7]
OW = 3.0
for i, (name, sub) in enumerate(OUTPUT_DATA):
    box(OX[i], 7.2, OW, 2.1, name, sub, fill="#E5EAED", edge=C_OUTPUT, fs=7.3)
    arr(AX[i] + AW/2, 10.65, OX[i] + OW/2, 9.3, color=C_OUTPUT, lw=1.0)

box(0.5, 6.45, 8.5, 0.65,
    "SkillResult(result_id, skill_id, params, text, table_data, chart_json)  stored in ConversationState.prior_results[]",
    fill="#E5EAED", edge=C_OUTPUT, fs=7.5)

box(9.3, 6.45, 8.2, 0.65,
    "Export: Export bar → export_to_dataset() → dataiku.Dataset.write_with_schema(df)  ·  Protocol PDF: FileDownload via reportlab",
    fill="#FFE0D6", edge=C_DATA, fs=7.3)

arr(9.0, 7.2, 1.95, 24.4, "rendered in\nChatInterface", color=C_PANEL, lw=1.2, rad=0.15)

# Data Reasoning reads prior_results
arr(14.5, 6.45, 15.6, 20.35, "prior_results[]\nas context", color=C_ORCH, lw=1.0, rad=0.3, dashed=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
LEGEND = [
    (C_NOTEBOOK, "Notebook Cells"),
    (C_PANEL,    "Panel UI"),
    (C_ORCH,     "Orchestrator / FSM"),
    (C_LLM,      "LLM Client & Prompts"),
    (C_AGENT,    "Skill Agents"),
    (C_DATA,     "Data / Config"),
    (C_OUTPUT,   "Outputs / Export"),
]
ax.text(0.5, 6.1, "LEGEND:", fontsize=8, fontweight="bold", color="#333")
for i, (col, lbl) in enumerate(LEGEND):
    lx = 0.5 + i * 2.45
    rect = FancyBboxPatch((lx, 5.65), 0.42, 0.24,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=col, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.55, 5.77, lbl, fontsize=7.5, va="center", color="#333")

ax.text(9, 5.3,
    "Flow: User types → chat_callback (periodic trace refresh) → process_message → FSM or data_reasoning bypass "
    "→ classify_intent → extract_params → confirm → agent.run → AgentResult → _response_to_panel → ChatInterface",
    ha="center", fontsize=7, color="#555", style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
out_pdf = "/Users/jeremyzhang/conv_analytics_prototype/conv_analytics_notebook_flow.pdf"
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=150)
plt.savefig(out_png, format="png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
