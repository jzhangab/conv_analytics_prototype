# Migration & Integration Guide
## Integrating `conv_analytics_prototype` into an Existing React + Python Application

**Audience:** Software engineers performing the integration  
**Scope:** Backend wrapper interface, LLM service swap, Snowflake data layer, session history persistence, Flask API contract, React frontend wiring

---

## Table of Contents
1. [Architecture Overview — Single Entry & Exit](#1-architecture-overview)
2. [Component Migration Map](#2-component-migration-map)
3. [Lift-and-Shift: The Backend Wrapper](#3-lift-and-shift-the-backend-wrapper)
4. [LLM Service Migration — Dataiku LLM Mesh → Standard API](#4-llm-service-migration)
5. [Snowflake Migration — Replacing Dataiku Dataset Access](#5-snowflake-migration)
6. [Session History Persistence — In-Memory → Persistent Store](#6-session-history-persistence)
7. [Flask API Contract — React Frontend Integration](#7-flask-api-contract)
8. [Frontend Integration Notes (React)](#8-frontend-integration-notes-react)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)
10. [Migration Checklist](#10-migration-checklist)

---

## 1. Architecture Overview

The backend exposes a **single entry point and single exit point** through the `ChatBackend` wrapper. Every interaction — text messages, file uploads, confirmations, exports — flows through one method call.

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│              POST /api/interact  (JSON or multipart)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │  ChatRequest
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  backend/api/chat_backend.py                     │
│                       ChatBackend.process()                      │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Orchestrator (FSM)                   │   │
│   │  IDLE → PARAMETER_GATHERING → CONFIRMATION_PENDING      │   │
│   │       → SKILL_EXECUTION → ANALYSIS_PLANNING             │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │   Router → SubAgents    │                        │
│              │  (8 skills — see §3)    │                        │
│              └─────────────────────────┘                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │  ChatResponse
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  message · fsm_state · table_data · chart_json                  │
│  downloadable_files (base64 CSV) · uploaded_file_metadata       │
└─────────────────────────────────────────────────────────────────┘
```

### What this means for integration

The entire backend can be lifted into the target application by importing two objects:

```python
from backend.api.chat_backend import ChatBackend
from backend.api.models import ChatRequest

backend = ChatBackend()   # one instance, process-lifetime
response = backend.process(ChatRequest(session_id="...", action="message", message="..."))
```

`webapp.py` is a thin HTTP adapter over this call. You can keep it, replace it, or skip it entirely and call `ChatBackend.process()` directly from your existing Python server.

---

## 2. Component Migration Map

```
conv_analytics_prototype               →   Target Application
──────────────────────────────────────────────────────────────────────
backend/api/chat_backend.py            →   KEEP — primary integration surface
  └─ ChatBackend.process(ChatRequest)      └─ call directly or via HTTP adapter

backend/api/models.py                  →   KEEP — shared contract types
  └─ ChatRequest, ChatResponse,            └─ no changes needed
     UploadedFile, DownloadableFile

webapp.py                              →   KEEP or REPLACE HTTP adapter
  └─ POST /api/interact (unified)          └─ wire to your existing Flask/FastAPI app
  └─ legacy /chat /upload /confirm         └─ drop once React is live
     /export routes

backend/llm/llm_client.py             →   REPLACE (§4)
  └─ Dataiku LLM Mesh calls               └─ Anthropic / OpenAI SDK

backend/agents/*_agent.py (8 agents)  →   MODIFY one method per data-loading agent (§5)
  └─ dataiku.Dataset().get_dataframe()     └─ snowflake_client.query_to_df(sql)

backend/state/session_store.py        →   REPLACE (§6)
  └─ in-memory dict + TTL eviction         └─ Snowflake-backed persistent store
```

### What transfers unchanged

| Module | Path |
|---|---|
| Orchestrator FSM | `backend/orchestrator/orchestrator.py` |
| Intent Classifier | `backend/orchestrator/intent_classifier.py` |
| Parameter Extractor | `backend/orchestrator/parameter_extractor.py` |
| Confirmation Manager | `backend/orchestrator/confirmation_manager.py` |
| Router | `backend/orchestrator/router.py` |
| All 8 Agent files | `backend/agents/*.py` (data-load isolated to one method each — §5) |
| Prompt Templates | `backend/llm/prompt_templates.py` |
| Response Parser | `backend/llm/response_parser.py` |
| Parameter Schema | `backend/state/parameter_schema.py` |
| Conversation State | `backend/state/conversation_state.py` |
| String Matching | `backend/utils/string_matching.py` |
| Chart Builder | `backend/utils/chart_builder.py` |
| Formatters / Validators | `backend/utils/` |
| Skills Config | `config/skills_config.yaml` |

**Three areas require changes:** LLM client, data access layer, session store.

---

## 3. Lift-and-Shift: The Backend Wrapper

### Entry: `ChatRequest`

All interactions are expressed as a `ChatRequest` dataclass (`backend/api/models.py`):

| Field | Type | Used by action |
|---|---|---|
| `session_id` | `str` | all |
| `action` | `"message" \| "confirm" \| "upload" \| "export"` | all |
| `message` | `str` | `message`, `confirm` |
| `confirmed` | `bool` | `confirm` (legacy confirm endpoint) |
| `edit_params` | `dict` | `confirm` with edits |
| `files` | `list[UploadedFile]` | `upload` |
| `result_id` | `str` | `export` |
| `export_destination` | `str` | `export` |

`UploadedFile` is a framework-agnostic file holder — it duck-types Werkzeug `FileStorage` so existing agent parsing code requires no changes:

```python
@dataclass
class UploadedFile:
    file_key: str          # "site_file" or "protocol_file"
    filename: str
    data: bytes
    content_type: str = "application/octet-stream"

    def read(self) -> bytes: return self.data
    def stream(self): return io.BytesIO(self.data)
```

### Exit: `ChatResponse`

Every `backend.process()` call returns a `ChatResponse`:

```python
@dataclass
class ChatResponse:
    session_id: str
    action: str
    success: bool
    message: str                    # assistant markdown text
    fsm_state: str                  # idle | parameter_gathering | confirmation_pending | ...
    active_skill: str | None        # skill currently in progress
    skill_id: str | None            # skill that just completed
    result_id: str | None           # UUID for this skill result
    table_data: list[dict] | None   # result rows
    table_columns: list[str] | None # column labels
    chart_json: dict | None         # Bokeh JSON item for chart rendering
    downloadable_files: list[DownloadableFile]  # base64-encoded CSV/Excel
    uploaded_file_metadata: dict | None         # info about uploaded file
    error: str | None
```

`DownloadableFile` carries base64-encoded content and metadata so files can be transported over JSON:

```python
@dataclass
class DownloadableFile:
    filename: str
    content_type: str
    data_base64: str    # base64-encoded bytes
    description: str = ""
```

When a skill produces `table_data`, the backend automatically generates a downloadable CSV in `downloadable_files` — no extra work needed.

### Calling from Python

```python
from backend.api.chat_backend import ChatBackend
from backend.api.models import ChatRequest, UploadedFile

# Initialize once — pass optional overrides for LLM config and Snowflake
backend = ChatBackend(
    llm_client=my_llm_client,      # optional: inject your own LLMClient
    session_store=my_session_store, # optional: inject persistent session store
    snowflake_client=my_sf_client,  # optional: inject Snowflake client
)

# Text message
resp = backend.process(ChatRequest(
    session_id="session-123",
    action="message",
    message="benchmark KRAS G12C in adults, Phase 3",
))
print(resp.message)       # assistant markdown
print(resp.fsm_state)     # "confirmation_pending" — bot is asking user to confirm params

# Confirm the parameters
resp = backend.process(ChatRequest(
    session_id="session-123",
    action="confirm",
    message="yes",
))
print(resp.table_data)    # skill result rows
print(resp.chart_json)    # Bokeh chart (if applicable)

# Upload a file
resp = backend.process(ChatRequest(
    session_id="session-123",
    action="upload",
    files=[UploadedFile(
        file_key="site_file",
        filename="sites.csv",
        data=open("sites.csv", "rb").read(),
    )],
))

# Export a result
resp = backend.process(ChatRequest(
    session_id="session-123",
    action="export",
    result_id="uuid-of-result",
    export_destination="MY_EXPORT_TABLE",
))
```

### The 8 Skills

| Skill ID | Agent | Primary Dataset |
|---|---|---|
| `cro_site_profiling` | `CROSiteProfilingAgent` | `CTMS_DATASET_JOIN_ISSUE_DATASET` |
| `trial_benchmarking` | `TrialBenchmarkingAgent` | `CITELINE_DATA` |
| `competitive_intelligence` | `CompetitiveIntelligenceAgent` | `CITELINE_DATA` |
| `drug_reimbursement` | `DrugReimbursementAgent` | Web search only |
| `enrollment_forecasting` | `EnrollmentForecastingAgent` | Web search only |
| `protocol_analysis` | `ProtocolAnalysisAgent` | Uploaded file |
| `country_ranking` | `CountryRankingAgent` | Web search only |
| `reforecasting` | `ReforecastingAgent` | `REFORECAST` |

`competitive_intelligence` shares the Citeline dataset with `trial_benchmarking` but filters for trials whose status is "not yet started" (or start year ≥ current year as fallback). No additional dataset is required.

---

## 4. LLM Service Migration

### What to Replace

`backend/llm/llm_client.py` wraps the Dataiku LLM Mesh API. The rest of the codebase only ever calls:

- `llm.complete(messages: list[dict], temperature: float) -> str`
- `llm.complete_json(messages: list[dict], temperature: float) -> dict`

Replace the class body while keeping the same public interface so no call sites change.

### Anthropic Claude

```python
# backend/llm/llm_client.py
import os, anthropic
from backend.llm.response_parser import ResponseParser

class LLMClient:
    def __init__(self, config: dict):
        self.client     = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model      = config.get("model", "claude-sonnet-4-6")
        self.max_tokens = config.get("max_tokens", 16384)
        self.temp_classify     = config.get("temperature_classify", 0.1)
        self.temp_extract      = config.get("temperature_extract", 0.1)
        self.temp_agents       = config.get("temperature_agents", 0.3)
        self.temp_deterministic = config.get("temperature_deterministic", 0.0)
        self.call_log: list[dict] = []

    def complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=temperature, system=system, messages=chat_messages,
        )
        result = response.content[0].text
        self.call_log.append({"messages": messages, "response": result})
        return result

    def complete_json(self, messages: list[dict], temperature: float = 0.1) -> dict:
        return ResponseParser.parse_json(self.complete(messages, temperature))
```

### OpenAI / Azure OpenAI

```python
# backend/llm/llm_client.py  (OpenAI variant)
import os, openai
from backend.llm.response_parser import ResponseParser

class LLMClient:
    def __init__(self, config: dict):
        self.client     = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model      = config.get("model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 16384)
        self.temp_classify      = config.get("temperature_classify", 0.1)
        self.temp_extract       = config.get("temperature_extract", 0.1)
        self.temp_agents        = config.get("temperature_agents", 0.3)
        self.temp_deterministic = config.get("temperature_deterministic", 0.0)
        self.call_log: list[dict] = []

    def complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=self.max_tokens,
        )
        result = response.choices[0].message.content
        self.call_log.append({"messages": messages, "response": result})
        return result

    def complete_json(self, messages: list[dict], temperature: float = 0.1) -> dict:
        return ResponseParser.parse_json(self.complete(messages, temperature))
```

### Temperature Config (`config/llm_config.yaml`)

```yaml
llm_mesh:
  model: "claude-sonnet-4-6"
  temperature_classify: 0.1
  temperature_extract: 0.1
  temperature_agents: 0.3
  temperature_deterministic: 0.0
  max_tokens: 16384
  context_window_turns: 10
```

---

## 5. Snowflake Migration

### Current Dataiku Access Pattern

Three agents read reference datasets; one endpoint writes back:

| Agent | Dataiku Dataset | Access pattern |
|---|---|---|
| `CROSiteProfilingAgent` | `CTMS_DATASET_JOIN_ISSUE_DATASET` | Full load, cached per process |
| `TrialBenchmarkingAgent` | `CITELINE_DATA` | Full load, cached per process |
| `CompetitiveIntelligenceAgent` | `CITELINE_DATA` | Full load, shared cache with benchmarking |
| `ReforecastingAgent` | `REFORECAST` | Full load, filtered in Python |
| `/export` endpoint | user-named dataset | Write result DataFrame |

### Snowflake Client

```python
# backend/data/snowflake_client.py
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

class SnowflakeClient:
    def __init__(self):
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.is_closed():
            self._conn = snowflake.connector.connect(
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
                database=os.environ["SNOWFLAKE_DATABASE"],
                schema=os.environ["SNOWFLAKE_SCHEMA"],
            )
        return self._conn

    def query_to_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        cur = self._get_conn().cursor()
        cur.execute(sql, params)
        return cur.fetch_pandas_all()

    def insert_df(self, df: pd.DataFrame, table_name: str, overwrite: bool = False) -> None:
        write_pandas(self._get_conn(), df, table_name.upper(),
                     auto_create_table=True, overwrite=overwrite)

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._get_conn().cursor().execute(sql, params)
```

Install: `pip install snowflake-connector-python[pandas]`

### Agent-Level Changes

Each agent isolates Dataiku access in exactly one method. Only those methods change.

#### CROSiteProfilingAgent

```python
# BEFORE
import dataiku
df = dataiku.Dataset("CTMS_DATASET_JOIN_ISSUE_DATASET").get_dataframe()

# AFTER — inject SnowflakeClient at construction
class CROSiteProfilingAgent(BaseAgent):
    def __init__(self, llm, snowflake_client):
        self.llm = llm
        self.sf  = snowflake_client

    def _load_ctms(self) -> pd.DataFrame:
        return self.sf.query_to_df("SELECT * FROM CTMS_SITES")
```

#### TrialBenchmarkingAgent & CompetitiveIntelligenceAgent

Both use the same Citeline dataset and the same load method (`_load_citeline_df`). Change it once in `TrialBenchmarkingAgent`; `CompetitiveIntelligenceAgent` inherits the change:

```python
# BEFORE
df = dataiku.Dataset("CITELINE_DATA").get_dataframe()

# AFTER
def _load_citeline_df(self):
    df = self.sf.query_to_df("SELECT * FROM CITELINE_TRIALS")
    return df, None
```

#### ReforecastingAgent

```python
# BEFORE
df = dataiku.Dataset("REFORECAST").get_dataframe()

# AFTER — filter at query time (more efficient than full load)
def _load_reforecast(self, protocol_id: str) -> pd.DataFrame:
    return self.sf.query_to_df(
        "SELECT * FROM REFORECAST_DATA WHERE PROTOCOL_ID = %s",
        (protocol_id,)
    )
```

#### Export via ChatBackend

The `export` action in `ChatBackend._handle_export()` already retrieves the result from session state and returns it as a `DownloadableFile`. To additionally write it to Snowflake, inject a `snowflake_client` when constructing `ChatBackend`:

```python
backend = ChatBackend(snowflake_client=SnowflakeClient())
```

Then in `ChatBackend._handle_export()`:

```python
if self._snowflake and result.table_data:
    df = pd.DataFrame(result.table_data, columns=result.table_columns)
    self._snowflake.insert_df(df, req.export_destination or "EXPORT_RESULTS")
```

### Router Update

```python
# backend/orchestrator/router.py
from backend.data.snowflake_client import SnowflakeClient

class Router:
    def __init__(self, llm, config=None, web_search=None, snowflake_client=None):
        sf = snowflake_client
        citeline_dataset = (config or {}).get("data_sources", {}).get("citeline_dataset", "CITELINE_DATA")
        self._registry = {
            "cro_site_profiling":      CROSiteProfilingAgent(llm, sf),
            "trial_benchmarking":      TrialBenchmarkingAgent(llm, sf, web_search=web_search),
            "competitive_intelligence": CompetitiveIntelligenceAgent(llm, sf, web_search=web_search),
            "drug_reimbursement":      DrugReimbursementAgent(llm, web_search=web_search),
            "enrollment_forecasting":  EnrollmentForecastingAgent(llm, web_search=web_search),
            "protocol_analysis":       ProtocolAnalysisAgent(llm, web_search=web_search),
            "country_ranking":         CountryRankingAgent(llm, web_search=web_search),
            "reforecasting":           ReforecastingAgent(sf),
        }
```

---

## 6. Session History Persistence

### Current State

`backend/state/session_store.py` uses an in-memory dict with TTL eviction. Sessions are lost on process restart.

### Snowflake Schema

```sql
CREATE TABLE chat_sessions (
    session_id      VARCHAR(64)   PRIMARY KEY,
    user_id         VARCHAR(255),
    created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
    last_activity   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
    fsm_state       VARCHAR(50)   DEFAULT 'idle',
    active_skill    VARCHAR(100),
    metadata        VARIANT       -- collected_params, pending_confirmation, etc.
);

CREATE TABLE chat_messages (
    message_id  VARCHAR(64)   DEFAULT UUID_STRING() PRIMARY KEY,
    session_id  VARCHAR(64)   REFERENCES chat_sessions(session_id),
    role        VARCHAR(20),
    content     TEXT,
    timestamp   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
    metadata    VARIANT
);

CREATE TABLE skill_results (
    result_id       VARCHAR(64) PRIMARY KEY,
    session_id      VARCHAR(64) REFERENCES chat_sessions(session_id),
    skill_id        VARCHAR(100),
    parameters_used VARIANT,
    text_response   TEXT,
    table_data      VARIANT,
    table_columns   VARIANT,
    chart_json      VARIANT,
    timestamp       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Persistent SessionStore

```python
# backend/state/session_store.py
import json, uuid
from backend.state.conversation_state import ConversationState, Message, SkillResult, FSMState
from backend.data.snowflake_client import SnowflakeClient

class SessionStore:
    def __init__(self, snowflake_client: SnowflakeClient):
        self.sf = snowflake_client
        self._cache: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str, user_id: str = None) -> ConversationState:
        if session_id in self._cache:
            return self._cache[session_id]
        state = self._load_from_db(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)
            self.sf.execute(
                "INSERT INTO chat_sessions (session_id, user_id) VALUES (%s, %s)",
                (session_id, user_id)
            )
        self._cache[session_id] = state
        return state

    def save(self, state: ConversationState) -> None:
        self._cache[state.session_id] = state
        meta = json.dumps({"collected_parameters": state.collected_parameters})
        self.sf.execute(
            """MERGE INTO chat_sessions t USING (SELECT %s sid) s ON t.session_id = s.sid
               WHEN MATCHED THEN UPDATE SET
                 fsm_state = %s, active_skill = %s,
                 last_activity = CURRENT_TIMESTAMP, metadata = PARSE_JSON(%s)""",
            (state.session_id, state.fsm_state.name.lower(), state.active_skill, meta)
        )

    def append_message(self, session_id: str, message: Message) -> None:
        self.sf.execute(
            """INSERT INTO chat_messages
               (message_id, session_id, role, content, timestamp, metadata)
               SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s)""",
            (str(uuid.uuid4()), session_id, message.role, message.content,
             message.timestamp.isoformat(), json.dumps(message.metadata or {}))
        )

    def append_skill_result(self, session_id: str, result: SkillResult) -> None:
        self.sf.execute(
            """INSERT INTO skill_results
               (result_id, session_id, skill_id, parameters_used, text_response,
                table_data, table_columns, chart_json, timestamp)
               SELECT %s,%s,%s,PARSE_JSON(%s),%s,PARSE_JSON(%s),PARSE_JSON(%s),PARSE_JSON(%s),%s""",
            (result.result_id, session_id, result.skill_id,
             json.dumps(result.parameters_used), result.text_response,
             json.dumps(result.table_data), json.dumps(result.table_columns),
             json.dumps(result.chart_json), result.timestamp.isoformat())
        )

    def list_sessions(self, user_id: str) -> list[dict]:
        df = self.sf.query_to_df(
            "SELECT session_id, created_at, last_activity FROM chat_sessions "
            "WHERE user_id = %s ORDER BY last_activity DESC", (user_id,)
        )
        return df.to_dict(orient="records")

    def get_history(self, session_id: str) -> list[dict]:
        df = self.sf.query_to_df(
            "SELECT role, content, timestamp FROM chat_messages "
            "WHERE session_id = %s ORDER BY timestamp ASC", (session_id,)
        )
        return df.to_dict(orient="records")
```

Inject into `ChatBackend`:

```python
backend = ChatBackend(
    session_store=SessionStore(SnowflakeClient()),
    snowflake_client=SnowflakeClient(),
)
```

---

## 7. Flask API Contract

`webapp.py` is the HTTP adapter. It translates HTTP requests into `ChatRequest` objects, calls `ChatBackend.process()`, and returns `ChatResponse.to_dict()` as JSON. No business logic lives in `webapp.py`.

### Unified Endpoint — `POST /api/interact`

This is the **primary endpoint** for the React frontend. All four action types go through it.

**Action: `message`** (JSON body)

```json
{ "session_id": "uuid", "action": "message", "message": "benchmark KRAS G12C adults Phase 3" }
```

**Action: `confirm`** (JSON body)

```json
{ "session_id": "uuid", "action": "confirm", "message": "yes" }
```

**Action: `upload`** (multipart/form-data)

```
session_id=uuid
file_key=site_file          (or "protocol_file")
site_file=<binary>
```

**Action: `export`** (JSON body)

```json
{ "session_id": "uuid", "action": "export", "result_id": "uuid", "export_destination": "MY_TABLE" }
```

### Response Shape (`ChatResponse.to_dict()`)

Every response from `/api/interact` has this shape:

```json
{
  "session_id": "uuid",
  "action": "message",
  "success": true,
  "message": "**Trial Benchmarking: KRAS G12C — Phase 3 — Adult**\n\n...",
  "fsm_state": "idle",
  "active_skill": null,
  "skill_id": "trial_benchmarking",
  "result_id": "uuid-of-result",
  "table_data": [{"Trial ID": "NCT001", "Phase": "Phase 3", ...}, ...],
  "table_columns": ["Trial ID", "Phase", "Sites", "Patients", ...],
  "chart_json": null,
  "downloadable_files": [
    {
      "filename": "trial_benchmarking_results.csv",
      "content_type": "text/csv",
      "data_base64": "VHJpYWwgSUQs...",
      "description": "Download results as CSV"
    }
  ],
  "uploaded_file_metadata": null,
  "error": null
}
```

**`fsm_state` values:**
- `idle` — ready for next request
- `parameter_gathering` — bot is collecting missing skill parameters
- `confirmation_pending` — all parameters collected, awaiting user confirmation
- `analysis_planning` — bot generated a plan, awaiting approval
- `skill_execution` — transient; should resolve before response is returned

### Legacy Routes (kept for backward compatibility)

`webapp.py` also exposes:
- `POST /chat` → `action="message"`
- `POST /upload` → `action="upload"`
- `POST /confirm` → `action="confirm"`
- `POST /export` → `action="export"`

These forward to `ChatBackend.process()` the same way as `/api/interact`. Keep them during transition; drop them once the React frontend is fully wired.

### Session History Endpoints (new — add to `webapp.py`)

```python
@app.route("/sessions")
def list_sessions():
    user_id = request.args.get("user_id")
    backend, err = _guard()
    if err: return jsonify(err[0]), err[1]
    sessions = backend.session_store.list_sessions(user_id)
    return jsonify({"sessions": sessions})

@app.route("/sessions/<session_id>/history")
def session_history(session_id):
    backend, err = _guard()
    if err: return jsonify(err[0]), err[1]
    messages = backend.session_store.get_history(session_id)
    return jsonify({"messages": messages})
```

### CORS

```python
from flask_cors import CORS
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","))
```

Already configured in `webapp.py`. Set `ALLOWED_ORIGINS` for production.

---

## 8. Frontend Integration Notes (React)

### API Layer — Single Endpoint

All calls target `/api/interact`. Create a typed client:

```typescript
// src/api/chatApi.ts
const BASE = process.env.REACT_APP_API_URL ?? "http://localhost:5000";

async function interact(body: object): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/interact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export const sendMessage = (sessionId: string, message: string) =>
  interact({ session_id: sessionId, action: "message", message });

export const confirm = (sessionId: string, message: string) =>
  interact({ session_id: sessionId, action: "confirm", message });

export const exportResult = (sessionId: string, resultId: string, dest: string) =>
  interact({ session_id: sessionId, action: "export", result_id: resultId, export_destination: dest });

export async function uploadFile(sessionId: string, fileKey: string, file: File): Promise<ChatResponse> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("action", "upload");
  fd.append("file_key", fileKey);
  fd.append(fileKey, file);
  const res = await fetch(`${BASE}/api/interact`, { method: "POST", body: fd });
  return res.json();
}

export const getSessionHistory = (sessionId: string) =>
  fetch(`${BASE}/sessions/${sessionId}/history`).then(r => r.json());
```

### TypeScript Types

```typescript
// src/api/types.ts
interface DownloadableFile {
  filename: string;
  content_type: string;
  data_base64: string;
  description: string;
}

interface ChatResponse {
  session_id: string;
  action: string;
  success: boolean;
  message: string;
  fsm_state: "idle" | "parameter_gathering" | "confirmation_pending" | "analysis_planning";
  active_skill: string | null;
  skill_id: string | null;
  result_id: string | null;
  table_data: Record<string, unknown>[] | null;
  table_columns: string[] | null;
  chart_json: object | null;
  downloadable_files: DownloadableFile[];
  uploaded_file_metadata: object | null;
  error: string | null;
}
```

### Session ID Management

```typescript
function getOrCreateSessionId(): string {
  let id = localStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("session_id", id);
  }
  return id;
}

function startNewSession(): string {
  const id = crypto.randomUUID();
  localStorage.setItem("session_id", id);
  return id;
}
```

### Confirmation Flow

When `fsm_state === "confirmation_pending"` the bot has all parameters and is asking the user to confirm:

```typescript
if (response.fsm_state === "confirmation_pending") {
  setConfirmationMode(true);   // render Yes / Edit buttons
}

// "Yes" button:
const resp = await confirm(sessionId, "yes");

// "Edit" button — let user retype the parameters:
setConfirmationMode(false);    // back to free-text input
```

### File Downloads

`downloadable_files` is automatically populated when a skill returns table data. Decode and trigger a browser download:

```typescript
function downloadFile(f: DownloadableFile) {
  const bytes = atob(f.data_base64);
  const blob = new Blob(
    [Uint8Array.from(bytes, c => c.charCodeAt(0))],
    { type: f.content_type }
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = f.filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

### Bokeh Charts

`chart_json` is a Bokeh JSON item. Load Bokeh from CDN and embed:

```html
<!-- public/index.html -->
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.x.x.min.js"></script>
```

```tsx
// src/components/BokehChart.tsx
declare const Bokeh: any;

export function BokehChart({ chartJson, chartId }: { chartJson: object; chartId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current && chartJson) {
      ref.current.innerHTML = "";
      Bokeh.embed.embed_item(chartJson, chartId);
    }
  }, [chartJson, chartId]);
  return <div id={chartId} ref={ref} />;
}
```

Usage:
```tsx
{response.chart_json && (
  <BokehChart chartJson={response.chart_json} chartId={`chart-${response.result_id}`} />
)}
```

### Table Rendering

```tsx
{response.table_data && response.table_columns && (
  <DataTable columns={response.table_columns} rows={response.table_data} />
)}
```

### File Uploads

Two upload types are accepted:
- `file_key: "site_file"` — CRO site list CSV/Excel (triggers `cro_site_profiling` skill)
- `file_key: "protocol_file"` — Protocol PDF/DOCX (triggers `protocol_analysis` skill)

```tsx
<input type="file" onChange={e => {
  const file = e.target.files?.[0];
  if (file) uploadFile(sessionId, "site_file", file).then(setResponse);
}} />
```

### Session History Page

```typescript
// List past sessions for a user
const { sessions } = await fetch(`${BASE}/sessions?user_id=${userId}`).then(r => r.json());

// Load a specific session
const { messages } = await getSessionHistory(selectedSessionId);
// Replay messages into chat window in order
// Re-render table_data / chart_json from stored skill_results
```

---

## 9. Configuration & Environment Variables

```bash
# LLM (pick one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Snowflake
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_USER=svc_chatapp
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=CLINICAL_ANALYTICS
SNOWFLAKE_SCHEMA=CHATAPP

# Flask
FLASK_SECRET_KEY=change-me-in-prod
ALLOWED_ORIGINS=https://your-react-app.com

# Web Search (optional)
SERPAPI_KEY=...

# React
REACT_APP_API_URL=https://your-flask-api.com
```

`config/llm_config.yaml`:

```yaml
llm_mesh:
  model: "claude-sonnet-4-6"
  temperature_classify: 0.1
  temperature_extract: 0.1
  temperature_agents: 0.3
  temperature_deterministic: 0.0
  max_tokens: 16384
  context_window_turns: 10

data_sources:
  ctms_dataset: "CTMS_DATASET_JOIN_ISSUE_DATASET"
  citeline_dataset: "CITELINE_DATA"
  reforecast_dataset: "REFORECAST"
```

---

## 10. Migration Checklist

### Backend

- [ ] Replace `backend/llm/llm_client.py` with Anthropic or OpenAI implementation (§4)
- [ ] Create `backend/data/snowflake_client.py` (§5)
- [ ] Create Snowflake tables: `chat_sessions`, `chat_messages`, `skill_results` (§6)
- [ ] Load reference data into Snowflake: CTMS sites, Citeline trials, Reforecast data (§5)
- [ ] Update `CROSiteProfilingAgent._load_ctms()` to call `snowflake_client.query_to_df()` (§5)
- [ ] Update `TrialBenchmarkingAgent._load_citeline_df()` to call Snowflake — `CompetitiveIntelligenceAgent` inherits the fix (§5)
- [ ] Update `ReforecastingAgent._load_reforecast()` to use parameterized Snowflake query (§5)
- [ ] Update `Router.__init__` to accept and pass `snowflake_client` to data-dependent agents (§5)
- [ ] Add Snowflake write to `ChatBackend._handle_export()` (§5)
- [ ] Replace `backend/state/session_store.py` with `SnowflakeSessionStore` (§6)
- [ ] Wire `session_store.append_message()` in orchestrator after each turn (§6)
- [ ] Wire `session_store.append_skill_result()` in orchestrator after each skill execution (§6)
- [ ] Add `GET /sessions` and `GET /sessions/<session_id>/history` to `webapp.py` (§7)
- [ ] Confirm `ALLOWED_ORIGINS` env var is set for production CORS (§7)
- [ ] Remove `dataiku` from `requirements.txt`; add `snowflake-connector-python[pandas]`, `anthropic` or `openai`, `flask-cors`
- [ ] Set all environment variables (§9)

### Frontend (React)

- [ ] Create `src/api/chatApi.ts` using `/api/interact` unified endpoint (§8)
- [ ] Define `ChatResponse` TypeScript interface (§8)
- [ ] Implement session ID generation and `localStorage` persistence (§8)
- [ ] Handle `confirmation_pending` state — render Yes/Edit buttons (§8)
- [ ] Handle `downloadable_files` — decode base64 and trigger browser download (§8)
- [ ] Add Bokeh JS to `public/index.html` and create `BokehChart` component (§8)
- [ ] Implement file upload for `site_file` and `protocol_file` (§8)
- [ ] Render `table_data` / `table_columns` using a table component (§8)
- [ ] Build session history page using `/sessions` and `/sessions/<id>/history` (§8)
- [ ] Set `REACT_APP_API_URL` in `.env.production` (§9)

### Testing

- [ ] Smoke-test each of the 8 skills end-to-end with Snowflake data loaded
- [ ] Verify session persists across process restart (Snowflake store)
- [ ] Verify `competitive_intelligence` skill correctly filters for not-yet-started trials
- [ ] Verify session history loads and replays correctly in React
- [ ] Verify Bokeh chart renders in React via `embed_item`
- [ ] Verify file upload parses correctly and populates session state
- [ ] Verify confirmation flow (yes/no/edit) end-to-end
- [ ] Verify `/export` writes a valid Snowflake table and returns a `DownloadableFile`
- [ ] Verify CORS headers are correct from React origin in production
