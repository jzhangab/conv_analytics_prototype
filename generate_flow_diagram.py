"""
Generate a PDF flow diagram for the conv_analytics_prototype project.
Webapp (Flask) version — not the Panel notebook.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ──────────────────────────────────────────────────────────
C_USER   = "#4A90D9"   # blue   — user-facing
C_ORCH   = "#7B68EE"   # purple — orchestrator / FSM
C_LLM    = "#E8A838"   # amber  — LLM / prompts
C_AGENT  = "#3CB371"   # green  — skill agents
C_DATA   = "#CD5C5C"   # red    — data / state
C_OUTPUT = "#708090"   # slate  — output / export
C_ARROW  = "#444444"
BG       = "#F8F9FA"
WHITE    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(22, 30))
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ── Helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel=None, color=WHITE, edgecolor="#333",
        fontsize=9, bold=False, radius=0.25):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.4,
                          zorder=3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, zorder=4)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=6.8, color="#555", zorder=4, style="italic")
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, zorder=4)


def section_header(ax, x, y, w, h, label, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=color, edgecolor=color, linewidth=0,
                          alpha=0.88, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=WHITE, zorder=3)


def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW, rad=0.0, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, fontsize=7, color="#333", zorder=6,
                ha="left", va="center",
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.8, pad=1))


def dashed_box(ax, x, y, w, h, color="#999"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor="none", edgecolor=color, linewidth=1.2,
                          linestyle="--", zorder=2)
    ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(11, 29.5, "Clinical Analytics Chatbot — Webapp Architecture & Request Flow",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#222")
ax.text(11, 29.1, "Dataiku DSS · Flask Webapp · LLM Mesh (Azure OpenAI) · Agentic AI",
        ha="center", va="center", fontsize=9, color="#555")

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 0 — USER INTERFACE  (y: 26.0–28.8)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 27.55, 21.4, 0.50, "① USER INTERFACE  (Browser → Flask Webapp)", C_USER)

box(ax, 0.5, 26.05, 3.2, 1.3, "Browser UI",
    "Chat · Upload · Results · Export", color="#D6E8FA", edgecolor=C_USER, bold=True)

routes = [
    ("GET /",        "index.html"),
    ("POST /chat",   "text message"),
    ("POST /upload", "site_file\nprotocol_file"),
    ("POST /confirm","yes / no /\nedit_params"),
    ("POST /export", "dataset_name\nresult_id"),
    ("GET /healthz", "startup check"),
]
route_xs = [4.1, 6.8, 9.5, 12.2, 14.9, 17.6]
for i, (route, sub) in enumerate(routes):
    box(ax, route_xs[i], 26.05, 2.5, 1.3, route, sub,
        color="#D6E8FA", edgecolor=C_USER, fontsize=8.5)
    arrow(ax, 2.1, 26.7, route_xs[i]+1.25, 27.3)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — ORCHESTRATOR  (y: 21.5–27.3)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 25.5, 21.4, 0.45, "② ORCHESTRATOR  (orchestrator.py)", C_ORCH)

# FSM states
dashed_box(ax, 0.5, 22.4, 13.5, 2.9, color=C_ORCH)
ax.text(1.0, 25.2, "Finite State Machine", fontsize=8.5, color=C_ORCH, fontweight="bold")

fsm_states = [
    ("IDLE",          "awaiting\nnew intent"),
    ("PARAMETER\nGATHERING",  "collecting\nrequired params"),
    ("CONFIRMATION\nPENDING", "user review\nbefore execution"),
    ("SKILL\nEXECUTION",      "running\nsubagent"),
    ("ANALYSIS\nPLANNING",    "confirm plan\nbefore reasoning"),
]
state_xs = [0.7, 3.2, 5.9, 8.7, 11.2]
for i, (state, sub) in enumerate(fsm_states):
    box(ax, state_xs[i], 23.5, 2.3, 1.6, state, sub,
        color="#EDE8FF", edgecolor=C_ORCH, fontsize=8)

# FSM transitions
transitions = [
    (state_xs[0]+2.3, 24.3, state_xs[1], 24.3, "intent\nmatched"),
    (state_xs[1]+2.3, 24.3, state_xs[2], 24.3, "params\ncomplete"),
    (state_xs[2]+2.3, 24.3, state_xs[3], 24.3, "user\nconfirmed"),
]
for x1, y1, x2, y2, lbl in transitions:
    arrow(ax, x1, y1, x2, y2, label=lbl, color=C_ORCH, lw=1.2)

# Back to IDLE from SKILL_EXECUTION
ax.annotate("", xy=(state_xs[0]+1.15, 23.5),
            xytext=(state_xs[3]+1.15, 23.5),
            arrowprops=dict(arrowstyle="-|>", color=C_ORCH, lw=1.1,
                            connectionstyle="arc3,rad=0.4"), zorder=5)
ax.text(5.3, 22.6, "result / cancel → IDLE", fontsize=6.5, color=C_ORCH,
        ha="center", style="italic")

# ANALYSIS_PLANNING transition label
ax.text(14.0, 24.3, "results\navailable", fontsize=6.5, color=C_ORCH, ha="center")
arrow(ax, state_xs[0]+1.15, 23.5, state_xs[4]+1.15, 24.3, color=C_ORCH, lw=1.0, rad=-0.25)

# Orchestrator functions
box(ax, 0.7, 22.1, 2.3, 1.2, "classify_intent()\n+ intent_classifier.py",
    color="#EDE8FF", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 3.2, 22.1, 2.5, 1.2, "extract_parameters()\n+ parameter_extractor.py",
    color="#EDE8FF", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 6.0, 22.1, 2.5, 1.2, "build_confirmation\n_prompt()",
    color="#EDE8FF", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 8.7, 22.1, 2.5, 1.2, "router.get_agent()\nagent.run()",
    color="#EDE8FF", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 11.3, 22.1, 2.4, 1.2, "_generate_plan()\n_handle_reasoning()",
    color="#EDE8FF", edgecolor=C_ORCH, fontsize=7.5)

for i in range(5):
    sx = state_xs[i] + 1.15
    arrow(ax, sx, 23.5, sx, 23.3, color=C_ORCH, lw=1.0)

# Session management
box(ax, 14.5, 23.4, 3.5, 1.3, "SessionStore",
    "session_id → ConversationState\n30-min TTL, in-memory",
    color="#F0EBF8", edgecolor=C_ORCH, fontsize=8)
box(ax, 14.5, 22.0, 3.5, 1.2, "ConversationState",
    "messages · params · uploaded_files\nprior_results · fsm_state",
    color="#F0EBF8", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 18.3, 23.4, 3.1, 1.3, "SkillSchema\n(skills_config.yaml)",
    "required / optional params\nlabels, types, choices",
    color="#F0EBF8", edgecolor=C_ORCH, fontsize=7.5)
box(ax, 18.3, 22.0, 3.1, 1.2, "SkillResult\n(result store)",
    "result_id · table_data\nchart_json · text_response",
    color="#F0EBF8", edgecolor=C_ORCH, fontsize=7.5)

arrow(ax, 14.5, 24.0, 14.0, 24.0, color=C_ORCH)
arrow(ax, 14.5, 22.6, 14.0, 22.6, color=C_ORCH)

# Route → Orchestrator entry
arrow(ax, 10.0, 26.05, 6.5, 25.95, color=C_USER, lw=1.3)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — LLM CLIENT & WEB SEARCH  (y: 18.5–21.8)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 21.75, 21.4, 0.42, "③ LLM CLIENT & WEB SEARCH  (llm_client.py · web_search.py)", C_LLM)

box(ax, 0.5, 20.2, 3.4, 1.3, "LLMClient.complete()",
    "messages list → resp.text", color="#FFF3D6", edgecolor=C_LLM, bold=True)
box(ax, 4.3, 20.2, 3.2, 1.3, "complete_json()\n_parse_json()",
    "strip fences → json.loads", color="#FFF3D6", edgecolor=C_LLM)
box(ax, 7.9, 20.2, 3.4, 1.3, "_repair_json()\n(truncation recovery)",
    "close open strings/arrays/objects", color="#FFF3D6", edgecolor=C_LLM)
box(ax, 11.7, 20.2, 3.4, 1.3, "WebSearchClient",
    "search() → snippets block\nused by agents + reasoning",
    color="#FFF8E0", edgecolor=C_LLM, fontsize=8)

box(ax, 0.5, 19.35, 10.7, 0.75, "Dataiku LLM Mesh  →  project.get_llm(connection_id)  ·  llm.new_completion()  ·  completion.with_max_output_tokens(16384)  ·  completion.execute()",
    color="#FFE8A0", edgecolor=C_LLM, fontsize=8)

for cx in [2.2, 5.9, 9.6, 13.4]:
    arrow(ax, cx, 20.2, cx, 20.1, color=C_LLM)

# Prompt templates
box(ax, 15.5, 19.3, 6.0, 2.2, "prompt_templates.py",
    "INTENT_CLASSIFIER · PARAMETER_EXTRACTOR\n"
    "SITE_MATCHING · TRIAL_BENCHMARKING\n"
    "DRUG_REIMBURSEMENT · ENROLLMENT_PARAMS\n"
    "ENROLLMENT_NARRATIVE · ANALYSIS_PLAN\n"
    "DATA_REASONING · GENERAL_KNOWLEDGE",
    color="#FFF3D6", edgecolor=C_LLM, fontsize=7.8, bold=True)

# Orchestrator → LLM
arrow(ax, 6.5, 22.1, 6.0, 21.75, color=C_LLM, lw=1.2)
arrow(ax, 15.5, 20.4, 15.2, 20.4, color=C_LLM)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — SKILL AGENTS  (y: 13.5–19.1)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 19.05, 21.4, 0.42, "④ SKILL AGENTS  (router.py → agents/)", C_AGENT)

# Router
box(ax, 0.5, 17.5, 2.8, 1.3, "Router",
    "skill_id → agent\ninstance", color="#D8F0E0", edgecolor=C_AGENT, bold=True)

AGENTS = [
    ("CRO Site\nProfiling",        "Jaro-Winkler match\nvs CTMS DB"),
    ("Trial\nBenchmarking",        "Citeline DB query\n+ web search"),
    ("Drug\nReimbursement",        "HTA/payer assess.\n+ web search"),
    ("Enrollment\nForecasting",    "2-stage: LLM params\n→ 3 scenario curves"),
    ("Protocol\nAnalysis",         "Study design review\nof uploaded PDF"),
    ("Country\nRanking",           "Trial experience\nby country"),
    ("Enrollment\nReforecasting",  "Update forecast from\nactual site data"),
]
agent_xs = [3.6, 6.3, 9.0, 11.7, 14.4, 17.1, 19.5]
agent_ws = [2.5, 2.5, 2.5, 2.5, 2.5, 2.2, 2.2]
for i, (name, sub) in enumerate(AGENTS):
    box(ax, agent_xs[i], 16.4, agent_ws[i], 2.3, name, sub,
        color="#D8F0E0", edgecolor=C_AGENT, fontsize=8, bold=False)
    arrow(ax, 3.3, 18.15, agent_xs[i]+agent_ws[i]/2, 18.7, color=C_AGENT, lw=1.0)

# Data sources row
DATA_SOURCES = [
    (3.6,  2.5, "CTMS_SITES.csv", "sponsor CTMS DB\nsite_id · name · country"),
    (6.3,  2.5, "Citeline DB",    "indication · phase\nage_group · metrics"),
    (9.0,  2.5, "Web Search",     "HTA data · payer\npolicies · drug info"),
    (11.7, 2.5, "Enrollment\nModel", "logistic ramp\n3 scenarios"),
    (14.4, 2.5, "Protocol\nPDF Upload", "study design\n(full text / PDF)"),
    (17.1, 4.6, "Citeline DB\n+ REFORECAST\nDataset", "actual site\nactivation data"),
]
for dx, dw, dname, dsub in DATA_SOURCES:
    box(ax, dx, 14.1, dw, 2.0, dname, dsub,
        color="#FFE8D6", edgecolor=C_DATA, fontsize=7.8)

for i, (dx, dw, _, _) in enumerate(DATA_SOURCES):
    ax_x = dx + dw/2
    # only draw arrow if agent and data source columns align
    arrow(ax, ax_x, 16.4, ax_x, 16.1, color=C_DATA, lw=1.0)

# response_parser
box(ax, 0.5, 14.1, 2.8, 2.0, "response_parser.py",
    "parse_site_matching\nparse_enrollment\ntable builders",
    color="#FFE8D6", edgecolor=C_DATA, fontsize=7.5)
arrow(ax, 1.9, 16.4, 1.9, 16.1, color=C_DATA, lw=1.0)

# Router ← Orchestrator
arrow(ax, 9.0, 22.1, 1.9, 18.8, color=C_AGENT, lw=1.2, rad=0.1)

# LLM used by agents
arrow(ax, 9.6, 19.35, 9.6, 19.05, color=C_LLM, lw=1.1)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4 — AGENT OUTPUTS  (y: 10.8–13.8)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 13.85, 21.4, 0.42, "⑤ AGENT OUTPUTS  →  AgentResult  →  SkillResult", C_OUTPUT)

OUTPUTS = [
    ("Site Match\nTable",     "Row · Uploaded Site\nCTMS Match · Confidence"),
    ("Benchmark\nJSON",       "key_metrics · patterns\nchallenges · summary"),
    ("Reimbursement\nJSON",   "country_assessments\nlikelihood · requirements"),
    ("Enrollment\nForecast",  "Bokeh chart (3 scenarios)\npeak sites · months"),
    ("Protocol\nReport",      "study design review\nrisks · recommendations"),
    ("Country\nRanking",      "ranked table\nexperience score"),
    ("Reforecast\nReport",    "updated curves\nvs actuals"),
]
out_xs = [0.5, 3.3, 6.1, 8.9, 11.7, 14.5, 17.3]
out_ws = [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 3.9]
for i, (name, sub) in enumerate(OUTPUTS):
    box(ax, out_xs[i], 11.2, out_ws[i], 2.3, name, sub,
        color="#E8ECF0", edgecolor=C_OUTPUT, fontsize=7.8)

for i, ax_x0 in enumerate(agent_xs):
    cx = ax_x0 + agent_ws[i]/2
    arrow(ax, cx, 16.4, cx, 13.5, color=C_OUTPUT, lw=1.0)

# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5 — RESPONSE & EXPORT  (y: 7.8–10.8)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 10.95, 21.4, 0.42, "⑥ RESPONSE SERIALISATION & EXPORT", C_OUTPUT)

box(ax, 0.5, 9.2, 5.5, 1.5, "_serialize_response()",
    "jsonify( message · table_data · chart_json\n· fsm_state · result_id · uploaded_files )",
    color="#E8ECF0", edgecolor=C_OUTPUT, fontsize=8, bold=True)
box(ax, 6.4, 9.2, 4.5, 1.5, "Frontend Renderer",
    "Markdown · DataFrame table\nBokeh chart embed · confirm dialog",
    color="#D6E8FA", edgecolor=C_USER, fontsize=8)
box(ax, 11.3, 9.2, 4.2, 1.5, "POST /export",
    "dataiku.Dataset(name)\n.write_with_schema(df)",
    color="#E8ECF0", edgecolor=C_OUTPUT, fontsize=8)
box(ax, 15.9, 9.2, 3.8, 1.5, "Dataiku Dataset",
    "persisted result\n(on user request only)",
    color="#FFE8D6", edgecolor=C_DATA, fontsize=8)

arrow(ax, 6.0, 9.95, 6.4, 9.95, color=C_OUTPUT)
arrow(ax, 10.9, 11.2, 8.65, 10.7, color=C_OUTPUT)
arrow(ax, 11.2, 9.95, 11.3, 9.95, color=C_OUTPUT)
arrow(ax, 15.5, 9.95, 15.9, 9.95, color=C_OUTPUT)

# JSON back to UI
arrow(ax, 2.75, 9.2, 2.0, 26.05, color=C_USER, lw=1.0, rad=0.35)
ax.text(0.6, 17.5, "JSON\nresponse\nback to UI", fontsize=6.5, color=C_USER,
        ha="center", rotation=90)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIRMATION LOOP
# ═══════════════════════════════════════════════════════════════════════════
ax.annotate("", xy=(route_xs[3]+1.25, 26.7), xytext=(7.15, 25.5),
            arrowprops=dict(arrowstyle="-|>", color=C_ORCH, lw=1.2,
                            connectionstyle="arc3,rad=-0.25"), zorder=5)
ax.text(11.5, 26.3, "confirmation\ndialog", fontsize=6.5, color=C_ORCH, ha="center")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG FILES  (y: 6.0–8.8)
# ═══════════════════════════════════════════════════════════════════════════
section_header(ax, 0.3, 8.95, 21.4, 0.40, "⑦ CONFIGURATION & SHARED PARAM INHERITANCE", "#888")

box(ax, 0.5, 7.4, 4.5, 1.3, "llm_config.yaml",
    "connection_id · max_tokens: 16384\ntemperatures · context_window_turns: 10",
    color="#F5F5F5", edgecolor="#888", fontsize=8)
box(ax, 5.4, 7.4, 4.5, 1.3, "skills_config.yaml",
    "7 skill schemas: required/optional params\nlabels, types, choices, data_type=file",
    color="#F5F5F5", edgecolor="#888", fontsize=8)
box(ax, 10.3, 7.4, 5.5, 1.3, "Shared Parameter Inheritance",
    "indication · age_group · phase\nauto-propagated across skill runs in a session",
    color="#F5F5F5", edgecolor="#888", fontsize=8)
box(ax, 16.2, 7.4, 5.3, 1.3, "File Upload Context",
    "site_file → CRO Site Profiling\nprotocol_file → Protocol Analysis",
    color="#F5F5F5", edgecolor="#888", fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND  (y: 5.2–7.0)
# ═══════════════════════════════════════════════════════════════════════════
ax.text(0.6, 6.95, "LEGEND", fontsize=8.5, fontweight="bold", color="#333")
legend_items = [
    (C_USER,   "User Interface / Flask routes"),
    (C_ORCH,   "Orchestrator / FSM"),
    (C_LLM,    "LLM Client & Prompts"),
    (C_AGENT,  "Skill Agents"),
    (C_DATA,   "Data / State"),
    (C_OUTPUT, "Output / Export"),
]
for i, (color, label) in enumerate(legend_items):
    col = i % 3
    row = i // 3
    lx = 0.6 + col * 6.8
    ly = 6.45 - row * 0.5
    rect = FancyBboxPatch((lx, ly), 0.5, 0.28,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.65, ly + 0.14, label, fontsize=8, va="center", color="#333")

ax.text(11, 5.75,
        "Call flow: User message  →  Orchestrator (FSM)  →  Intent Classify  →  "
        "Parameter Extract  →  Confirm  →  Router  →  Agent.run()  →  SkillResult  →  JSON response  →  UI",
        ha="center", fontsize=7.5, color="#555", style="italic")

ax.text(11, 5.35,
        "Data Reasoning flow: Prior results in session  →  Analysis Plan (user confirms)  →  "
        "DATA_REASONING prompt + web search  →  Grounded answer  →  UI",
        ha="center", fontsize=7.5, color="#555", style="italic")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
out = "/Users/jeremyzhang/conv_analytics_prototype/conv_analytics_flow_diagram.pdf"
plt.savefig(out, format="pdf", bbox_inches="tight", dpi=150)
plt.savefig(out.replace(".pdf", ".png"), format="png", bbox_inches="tight", dpi=150)
print(f"Saved: {out}")
print(f"Saved: {out.replace('.pdf', '.png')}")
