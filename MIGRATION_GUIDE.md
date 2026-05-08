# Migration & Integration Guide
## Integrating `conv_analytics_prototype` into an Existing React + Python Application

**Audience:** Software engineers performing the integration  
**Scope:** Orchestration, LLM services, agents/skills, Snowflake backend, session history persistence, React frontend API contract

---

## Table of Contents
1. [What Transfers Unchanged](#1-what-transfers-unchanged)
2. [Component Migration Map](#2-component-migration-map)
3. [LLM Service Migration — Dataiku LLM Mesh → Standard API](#3-llm-service-migration)
4. [Snowflake Migration — Replacing Dataiku Dataset Access](#4-snowflake-migration)
5. [Session History Persistence — In-Memory → Persistent Store](#5-session-history-persistence)
6. [Flask API Contract — React Frontend Integration](#6-flask-api-contract)
7. [Frontend Integration Notes (React)](#7-frontend-integration-notes-react)
8. [Configuration & Environment Variables](#8-configuration--environment-variables)
9. [Migration Checklist](#9-migration-checklist)

---

## 1. What Transfers Unchanged

The following modules contain no Dataiku-specific dependencies and can be dropped into the target application as-is:

| Module | Path | Notes |
|---|---|---|
| Orchestrator | `backend/orchestrator/orchestrator.py` | Pure Python FSM logic |
| Intent Classifier | `backend/orchestrator/intent_classifier.py` | Calls `llm_client` only |
| Parameter Extractor | `backend/orchestrator/parameter_extractor.py` | Calls `llm_client` only |
| Confirmation Manager | `backend/orchestrator/confirmation_manager.py` | Pure string logic |
| Router | `backend/orchestrator/router.py` | Only wires agents together |
| All 7 Agent files | `backend/agents/*.py` | Data-access calls isolated to one method each — see §4 |
| Prompt Templates | `backend/llm/prompt_templates.py` | Pure string constants |
| Response Parser | `backend/llm/response_parser.py` | Pure JSON parsing |
| Parameter Schema | `backend/state/parameter_schema.py` | Pure dataclasses |
| Conversation State | `backend/state/conversation_state.py` | Pure dataclasses |
| String Matching | `backend/utils/string_matching.py` | Pure Python |
| Chart Builder | `backend/utils/chart_builder.py` | Bokeh only |
| Formatters | `backend/utils/formatters.py` | Pure Python |
| Validators | `backend/utils/validators.py` | Pure Python |
| Skills Config | `config/skills_config.yaml` | No changes needed |

**Three areas require changes:** LLM client, data access layer, and session store.

---

## 2. Component Migration Map

```
conv_analytics_prototype          →   Target Application
─────────────────────────────────────────────────────────────────
backend/llm/llm_client.py         →   REPLACE (§3)
  └─ dataiku LLM Mesh calls           └─ Anthropic / OpenAI SDK calls

backend/agents/*_agent.py         →   MODIFY one method per agent (§4)
  └─ dataiku.Dataset().get_dataframe()  └─ snowflake_client.query_to_df(sql)

webapp.py → /export endpoint      →   MODIFY (§4)
  └─ dataiku.Dataset().write_with_schema()  └─ snowflake_client.insert_df(df, table)

backend/state/session_store.py    →   REPLACE (§5)
  └─ in-memory dict + TTL eviction      └─ Snowflake (or Postgres) session tables

webapp.py (Flask routes)          →   KEEP routes, add CORS, merge into app (§6)
  └─ Jinja2 template rendering         └─ remove; React serves its own HTML

frontend/ (HTML + vanilla JS)     →   REPLACE with React components (§7)
```

---

## 3. LLM Service Migration

### What to Replace
`backend/llm/llm_client.py` currently wraps the Dataiku LLM Mesh API. The rest of the codebase only calls two methods on it:

- `llm.complete(messages: list[dict], temperature: float) -> str`
- `llm.complete_json(messages: list[dict], temperature: float) -> dict`

Replace the class body while keeping the same public interface so no call sites need to change.

### Anthropic Claude Implementation

```python
# backend/llm/llm_client.py
import os, json, anthropic
from backend.llm.response_parser import ResponseParser

class LLMClient:
    def __init__(self, config: dict):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = config.get("model", "claude-sonnet-4-6")
        self.max_tokens = config.get("max_tokens", 16384)
        self.call_log: list[dict] = []

    def complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        # Split off a leading system message if present
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=chat_messages,
        )
        result = response.content[0].text
        self.call_log.append({"messages": messages, "response": result})
        return result

    def complete_json(self, messages: list[dict], temperature: float = 0.1) -> dict:
        raw = self.complete(messages, temperature)
        return ResponseParser.parse_json(raw)
```

### OpenAI / Azure OpenAI Implementation

```python
# backend/llm/llm_client.py  (OpenAI variant)
import os, openai
from backend.llm.response_parser import ResponseParser

class LLMClient:
    def __init__(self, config: dict):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = config.get("model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 16384)
        self.call_log: list[dict] = []

    def complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        result = response.choices[0].message.content
        self.call_log.append({"messages": messages, "response": result})
        return result

    def complete_json(self, messages: list[dict], temperature: float = 0.1) -> dict:
        raw = self.complete(messages, temperature)
        return ResponseParser.parse_json(raw)
```

### Temperature Config (`config/llm_config.yaml`)

```yaml
llm_mesh:
  model: "claude-sonnet-4-6"          # or "gpt-4o"
  temperature_classify: 0.1
  temperature_extract: 0.1
  temperature_agents: 0.3
  temperature_deterministic: 0.0
  max_tokens: 16384
  context_window_turns: 10
```

No changes needed to callers — `llm_config.yaml` is loaded once in `webapp.py` and passed to `LLMClient.__init__`.

---

## 4. Snowflake Migration

### Dataiku Access Pattern (Current)

Three agents load reference data from Dataiku datasets:

| Agent | Dataiku Dataset | Purpose |
|---|---|---|
| `CROSiteProfilingAgent` | `CTMS_DATASET` | Master site list for fuzzy matching |
| `TrialBenchmarkingAgent` | `CITELINE_DATA` | Historical trial statistics |
| `ReforecastingAgent` | `REFORECAST` | Protocol-level forecast data |

The `/export` endpoint writes results back to a user-named Dataiku dataset.

Each access is a single call: `dataiku.Dataset(name).get_dataframe()` or `dataiku.Dataset(name).write_with_schema(df)`.

### Snowflake Client

Create a shared Snowflake connector module:

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
        write_pandas(
            self._get_conn(),
            df,
            table_name.upper(),
            auto_create_table=True,
            overwrite=overwrite,
        )

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._get_conn().cursor().execute(sql, params)
```

Install: `pip install snowflake-connector-python[pandas]`

### Agent-Level Changes

Each agent has exactly one data-loading method. Only those methods need to change.

#### CROSiteProfilingAgent (`site_list_merger_agent.py`)

```python
# BEFORE
import dataiku
df = dataiku.Dataset("CTMS_DATASET").get_dataframe()

# AFTER — inject snowflake_client at construction
class CROSiteProfilingAgent(BaseAgent):
    def __init__(self, llm, snowflake_client):
        self.llm = llm
        self.sf = snowflake_client

    def _load_ctms(self) -> pd.DataFrame:
        return self.sf.query_to_df("SELECT * FROM CTMS_SITES")
```

Table name `CTMS_SITES` should match what the Snowflake DBA creates when migrating the Dataiku dataset.

#### TrialBenchmarkingAgent (`trial_benchmarking_agent.py`)

```python
# BEFORE
df = dataiku.Dataset("CITELINE_DATA").get_dataframe()

# AFTER
def _load_citeline(self) -> pd.DataFrame:
    return self.sf.query_to_df("SELECT * FROM CITELINE_TRIALS")
```

The agent already falls back to a local CSV if loading fails — keep that fallback for dev/test.

#### ReforecastingAgent (`reforecasting_agent.py`)

```python
# BEFORE
df = dataiku.Dataset("REFORECAST").get_dataframe()

# AFTER
def _load_reforecast(self, protocol_id: str) -> pd.DataFrame:
    return self.sf.query_to_df(
        "SELECT * FROM REFORECAST_DATA WHERE PROTOCOL_ID = %s",
        (protocol_id,)
    )
```

Filtering at the SQL layer is more efficient than loading all rows and filtering in Python.

#### Export Endpoint (`webapp.py`)

```python
# BEFORE
@app.route("/export", methods=["POST"])
def export():
    dataset_name = request.json.get("dataset_name")
    dataiku.Dataset(dataset_name).write_with_schema(df)

# AFTER
@app.route("/export", methods=["POST"])
def export():
    data = request.json
    table_name = data.get("table_name", "EXPORT_RESULTS")
    result_id = data.get("result_id")
    state = session_store.get(data["session_id"])
    result = next(r for r in state.prior_results if r.result_id == result_id)
    df = pd.DataFrame(result.table_data, columns=result.table_columns)
    snowflake_client.insert_df(df, table_name, overwrite=data.get("overwrite", False))
    return jsonify({"success": True, "table": table_name})
```

### Router Update (`backend/orchestrator/router.py`)

Pass `snowflake_client` when constructing data-dependent agents:

```python
from backend.data.snowflake_client import SnowflakeClient

def build_router(llm, snowflake_client: SnowflakeClient):
    return {
        "cro_site_profiling":     CROSiteProfilingAgent(llm, snowflake_client),
        "trial_benchmarking":     TrialBenchmarkingAgent(llm, snowflake_client),
        "drug_reimbursement":     DrugReimbursementAgent(llm),
        "enrollment_forecasting": EnrollmentForecastingAgent(llm),
        "protocol_analysis":      ProtocolAnalysisAgent(llm),
        "country_ranking":        CountryRankingAgent(llm),
        "reforecasting":          ReforecastingAgent(snowflake_client),
    }
```

---

## 5. Session History Persistence

### Current State (In-Memory)

`backend/state/session_store.py` stores all sessions in a Python dict with a threading lock and TTL eviction. Sessions are lost on process restart — there is no persistence today.

### What "Session History" Requires

The existing `ConversationState` already tracks everything needed:
- `messages` — full conversation turn-by-turn (`role`, `content`, `timestamp`)
- `prior_results` — completed skill runs with parameters, text, table data, chart JSON
- `fsm_state`, `active_skill`, `collected_parameters` — in-flight state
- `uploaded_files` — uploaded file metadata

Persistence means serializing these to a database on every write and rehydrating them on session resume.

### Snowflake Schema

```sql
-- Session metadata
CREATE TABLE chat_sessions (
    session_id      VARCHAR(64)     PRIMARY KEY,
    user_id         VARCHAR(255),
    created_at      TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    last_activity   TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    fsm_state       VARCHAR(50)     DEFAULT 'idle',
    active_skill    VARCHAR(100),
    metadata        VARIANT                          -- JSONB for collected_params, pending_confirmation, etc.
);

-- One row per message (append-only)
CREATE TABLE chat_messages (
    message_id      VARCHAR(64)     DEFAULT UUID_STRING() PRIMARY KEY,
    session_id      VARCHAR(64)     REFERENCES chat_sessions(session_id),
    role            VARCHAR(20),                     -- user | assistant | system
    content         TEXT,
    timestamp       TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    metadata        VARIANT                          -- optional: intent, skill_id, etc.
);

-- One row per completed skill execution
CREATE TABLE skill_results (
    result_id       VARCHAR(64)     PRIMARY KEY,
    session_id      VARCHAR(64)     REFERENCES chat_sessions(session_id),
    skill_id        VARCHAR(100),
    parameters_used VARIANT,
    text_response   TEXT,
    table_data      VARIANT,
    table_columns   VARIANT,
    chart_json      VARIANT,
    timestamp       TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP
);
```

Snowflake `VARIANT` columns store arbitrary JSON — ideal for `table_data`, `parameters_used`, and `chart_json`.

### Persistent Session Store

Replace `backend/state/session_store.py` with a Snowflake-backed implementation:

```python
# backend/state/session_store.py
import json, uuid
from datetime import datetime
from backend.state.conversation_state import ConversationState, Message, SkillResult, FSMState
from backend.data.snowflake_client import SnowflakeClient

class SessionStore:
    def __init__(self, snowflake_client: SnowflakeClient):
        self.sf = snowflake_client
        self._cache: dict[str, ConversationState] = {}    # process-local cache

    def get_or_create(self, session_id: str, user_id: str = None) -> ConversationState:
        if session_id in self._cache:
            return self._cache[session_id]
        state = self._load_from_db(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)
            self._create_session_row(session_id, user_id)
        self._cache[session_id] = state
        return state

    def save(self, state: ConversationState) -> None:
        """Call after every FSM transition."""
        self._cache[state.session_id] = state
        self._upsert_session(state)

    def append_message(self, session_id: str, message: Message) -> None:
        self.sf.execute(
            """INSERT INTO chat_messages (message_id, session_id, role, content, timestamp, metadata)
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
            "WHERE user_id = %s ORDER BY last_activity DESC",
            (user_id,)
        )
        return df.to_dict(orient="records")

    def get_history(self, session_id: str) -> list[dict]:
        df = self.sf.query_to_df(
            "SELECT role, content, timestamp FROM chat_messages "
            "WHERE session_id = %s ORDER BY timestamp ASC",
            (session_id,)
        )
        return df.to_dict(orient="records")

    # ── private helpers ────────────────────────────────────────────────────

    def _load_from_db(self, session_id: str) -> ConversationState | None:
        df = self.sf.query_to_df(
            "SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,)
        )
        if df.empty:
            return None
        row = df.iloc[0]
        state = ConversationState(session_id=session_id)
        state.fsm_state = FSMState[row["FSM_STATE"].upper()]
        state.active_skill = row.get("ACTIVE_SKILL")
        meta = json.loads(row["METADATA"]) if row["METADATA"] else {}
        state.collected_parameters = meta.get("collected_parameters", {})
        state.messages = self._load_messages(session_id)
        state.prior_results = self._load_skill_results(session_id)
        return state

    def _load_messages(self, session_id: str) -> list[Message]:
        df = self.sf.query_to_df(
            "SELECT role, content, timestamp, metadata FROM chat_messages "
            "WHERE session_id = %s ORDER BY timestamp ASC", (session_id,)
        )
        return [
            Message(role=r["ROLE"], content=r["CONTENT"],
                    timestamp=r["TIMESTAMP"], metadata=json.loads(r["METADATA"] or "{}"))
            for _, r in df.iterrows()
        ]

    def _load_skill_results(self, session_id: str) -> list[SkillResult]:
        df = self.sf.query_to_df(
            "SELECT * FROM skill_results WHERE session_id = %s ORDER BY timestamp ASC",
            (session_id,)
        )
        results = []
        for _, r in df.iterrows():
            results.append(SkillResult(
                result_id=r["RESULT_ID"], skill_id=r["SKILL_ID"],
                parameters_used=json.loads(r["PARAMETERS_USED"] or "{}"),
                text_response=r["TEXT_RESPONSE"],
                table_data=json.loads(r["TABLE_DATA"] or "null"),
                table_columns=json.loads(r["TABLE_COLUMNS"] or "null"),
                chart_json=json.loads(r["CHART_JSON"] or "null"),
                timestamp=r["TIMESTAMP"],
            ))
        return results

    def _create_session_row(self, session_id: str, user_id: str) -> None:
        self.sf.execute(
            "INSERT INTO chat_sessions (session_id, user_id) VALUES (%s, %s)",
            (session_id, user_id)
        )

    def _upsert_session(self, state: ConversationState) -> None:
        meta = json.dumps({
            "collected_parameters": state.collected_parameters,
            "pending_confirmation": None,   # serialize if needed
        })
        self.sf.execute(
            """MERGE INTO chat_sessions t USING (SELECT %s sid) s ON t.session_id = s.sid
               WHEN MATCHED THEN UPDATE SET
                 fsm_state = %s, active_skill = %s,
                 last_activity = CURRENT_TIMESTAMP, metadata = PARSE_JSON(%s)""",
            (state.session_id, state.fsm_state.name.lower(), state.active_skill, meta)
        )
```

**Orchestrator Integration:** After every `process_message` call, the orchestrator calls `session_store.save(state)`. The `append_message` and `append_skill_result` hooks should be called inside `_execute_skill` when results arrive.

---

## 6. Flask API Contract

These are the endpoints the React frontend must call. The existing routes stay unchanged — only remove the Jinja2 template render on `GET /` since React handles routing.

### `POST /chat`

**Purpose:** Send a user message; receive assistant response.

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "benchmark the KRAS G12C indication in adults phase 3"
}
```

**Response:**
```json
{
  "message": "<assistant markdown text>",
  "fsm_state": "confirmation_pending",
  "active_skill": "trial_benchmarking",
  "skill_id": "trial_benchmarking",
  "result_id": "uuid-if-skill-just-executed",
  "table_data": [{"Indication": "KRAS G12C", ...}, ...],
  "table_columns": ["Indication", "Phase", "Median Sites", ...],
  "chart_json": null,
  "uploaded_files": {},
  "error": null
}
```

**States returned in `fsm_state`:**
- `idle` — ready for next request
- `parameter_gathering` — bot is asking follow-up questions for missing params
- `confirmation_pending` — bot has all params and is asking user to confirm before running
- `analysis_planning` — bot has generated an analysis plan and is asking for approval
- `skill_execution` — (transient, should not persist in response)

### `POST /confirm`

**Purpose:** Send the user's yes/no/edit reply to a confirmation prompt.

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "yes"
}
```

**Response:** Same shape as `/chat`.

### `POST /upload`

**Purpose:** Upload a CSV or Excel file (site list, protocol PDF).

**Request:** `multipart/form-data`
```
file_key:     "site_file" | "protocol_file"
file:         <binary>
session_id:   "uuid-string"
```

**Response:**
```json
{
  "success": true,
  "filename": "sites.csv",
  "columns": ["Site Name", "City", "Address"],
  "file_key": "site_file"
}
```

### `POST /export`

**Purpose:** Write a skill result to Snowflake.

**Request:**
```json
{
  "session_id": "uuid-string",
  "result_id": "uuid-of-skill-result",
  "table_name": "MY_EXPORT_TABLE",
  "overwrite": false
}
```

**Response:**
```json
{
  "success": true,
  "table": "MY_EXPORT_TABLE"
}
```

### `GET /sessions?user_id=<id>`

**New endpoint** — not in current codebase, needs to be added for session history browsing.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "created_at": "2025-01-15T10:00:00Z",
      "last_activity": "2025-01-15T10:45:00Z"
    }
  ]
}
```

### `GET /sessions/<session_id>/history`

**New endpoint** — returns full message history for a session.

**Response:**
```json
{
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "skill_results": [
    {
      "result_id": "uuid",
      "skill_id": "trial_benchmarking",
      "parameters_used": {},
      "table_data": [...],
      "table_columns": [...],
      "chart_json": null,
      "timestamp": "..."
    }
  ]
}
```

### CORS Setup

Since React (port 3000) and Flask (port 5000) are separate origins, add:

```python
from flask_cors import CORS
CORS(app, origins=["http://localhost:3000", "https://your-prod-domain.com"])
```

Install: `pip install flask-cors`

### `GET /`

Remove or keep as a health redirect. React serves its own `index.html`.

---

## 7. Frontend Integration Notes (React)

### API Layer

Create a typed API client in React to match the Flask contract:

```typescript
// src/api/chatApi.ts
const BASE = process.env.REACT_APP_API_URL ?? "http://localhost:5000";

export async function sendMessage(sessionId: string, message: string) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return res.json();   // shape: ChatResponse (see §6)
}

export async function confirm(sessionId: string, message: string) {
  const res = await fetch(`${BASE}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return res.json();
}

export async function uploadFile(sessionId: string, fileKey: string, file: File) {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("file_key", fileKey);
  fd.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return res.json();
}

export async function exportResult(sessionId: string, resultId: string, tableName: string) {
  const res = await fetch(`${BASE}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, result_id: resultId, table_name: tableName }),
  });
  return res.json();
}

export async function getSessionHistory(sessionId: string) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/history`);
  return res.json();
}
```

### Session ID Management

The existing backend creates sessions lazily when the first `/chat` call arrives with an unknown `session_id`. React should:
1. Generate a `session_id` with `crypto.randomUUID()` on new chat start.
2. Persist it in `localStorage` to survive page refreshes.
3. On "New Chat" action, clear localStorage and generate a new ID.
4. Pass the stored ID on every API call.

```typescript
function getOrCreateSessionId(): string {
  let id = localStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("session_id", id);
  }
  return id;
}
```

### Confirmation Flow

The existing backend sets `fsm_state: "confirmation_pending"` when it needs the user to confirm parameters. React must detect this and render a confirmation UI:

```typescript
// In your chat reducer / component:
if (response.fsm_state === "confirmation_pending") {
  // Show inline yes/no/edit buttons instead of free-text input
  setConfirmationMode(true);
}

// On "Yes" button click:
await confirm(sessionId, "yes");

// On "Edit" button click:
setConfirmationMode(false);   // Let user type free-form edit request
```

The current vanilla JS implementation in `frontend/static/js/confirm_dialog.js` is a direct reference for this behaviour.

### Bokeh Charts in React

The backend returns Bokeh chart data as `chart_json` in the response — a Bokeh JSON item produced by `json_item(figure)`. Rendering requires the Bokeh JS library:

**Option 1 — CDN script tag in `public/index.html`:**
```html
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.x.x.min.js"></script>
```

**Option 2 — React component wrapping Bokeh embed:**
```tsx
// src/components/BokehChart.tsx
import { useEffect, useRef } from "react";

declare const Bokeh: any;   // Bokeh loaded via CDN

interface Props { chartJson: object; chartId: string; }

export function BokehChart({ chartJson, chartId }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && chartJson) {
      ref.current.innerHTML = "";    // clear previous render
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

### File Upload UI

React file inputs must `POST` to `/upload` as `multipart/form-data`. The response returns `columns` so the UI can confirm what was parsed. Two upload slots exist:
- `file_key: "site_file"` — CRO site list CSV/Excel
- `file_key: "protocol_file"` — Protocol PDF/DOCX

### Table Rendering

`table_data` is a `list[dict]` and `table_columns` is a `list[str]`. Use any React table library (TanStack Table, AG Grid, etc.) to render:

```tsx
{response.table_data && (
  <DataTable columns={response.table_columns} rows={response.table_data} />
)}
```

### Session History Page

Use `GET /sessions?user_id=<id>` to list prior sessions, then `GET /sessions/<id>/history` to reload a past conversation. On reload:
1. Set `session_id` in localStorage to the selected session ID.
2. Replay `messages` into the chat window in order.
3. Re-render any `skill_results` that have `table_data` or `chart_json`.

---

## 8. Configuration & Environment Variables

Remove all Dataiku environment variables. Add:

```bash
# LLM
ANTHROPIC_API_KEY=sk-ant-...          # if using Claude
OPENAI_API_KEY=sk-...                  # if using OpenAI

# Snowflake
SNOWFLAKE_ACCOUNT=xy12345.us-east-1   # <account>.<region>
SNOWFLAKE_USER=svc_chatapp
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=CLINICAL_ANALYTICS
SNOWFLAKE_SCHEMA=CHATAPP

# Web Search (optional)
SERPAPI_KEY=...
```

Update `config/llm_config.yaml`:

```yaml
llm_mesh:
  model: "claude-sonnet-4-6"
  # ... temperatures unchanged

data_sources:
  ctms_table: "CTMS_SITES"
  citeline_table: "CITELINE_TRIALS"
  reforecast_table: "REFORECAST_DATA"

serp_api:
  api_key: "${SERPAPI_KEY}"
  enabled: false
```

---

## 9. Migration Checklist

### Backend

- [ ] Replace `backend/llm/llm_client.py` with Anthropic or OpenAI implementation (§3)
- [ ] Create `backend/data/snowflake_client.py` (§4)
- [ ] Create Snowflake tables: `chat_sessions`, `chat_messages`, `skill_results`, and the three reference data tables (§4, §5)
- [ ] Load reference data (CTMS, Citeline, Reforecast) into Snowflake
- [ ] Update `CROSiteProfilingAgent.__init__` to accept `snowflake_client` (§4)
- [ ] Update `TrialBenchmarkingAgent._load_citeline()` to use Snowflake (§4)
- [ ] Update `ReforecastingAgent._load_reforecast()` to use Snowflake with parameterized query (§4)
- [ ] Update `/export` endpoint in `webapp.py` to write to Snowflake (§4)
- [ ] Replace `backend/state/session_store.py` with persistent implementation (§5)
- [ ] Wire `session_store.append_message()` calls in orchestrator (§5)
- [ ] Wire `session_store.append_skill_result()` calls in orchestrator after skill execution (§5)
- [ ] Add `GET /sessions` and `GET /sessions/<id>/history` endpoints to `webapp.py` (§6)
- [ ] Add `flask-cors` and configure allowed origins (§6)
- [ ] Update `build_router()` in `router.py` to pass `snowflake_client` (§4)
- [ ] Remove `dataiku` from `requirements.txt`; add `snowflake-connector-python[pandas]`, `anthropic` or `openai`, `flask-cors`
- [ ] Set environment variables (§8)

### Frontend (React)

- [ ] Create `src/api/chatApi.ts` typed API client (§7)
- [ ] Implement session ID generation and localStorage persistence (§7)
- [ ] Implement `confirmation_pending` state detection and Yes/No/Edit UI (§7)
- [ ] Add Bokeh JS to `public/index.html` and create `BokehChart` component (§7)
- [ ] Implement file upload for `site_file` and `protocol_file` (§7)
- [ ] Implement table rendering for `table_data` / `table_columns` (§7)
- [ ] Build session history page using `GET /sessions` and `GET /sessions/<id>/history` (§7)
- [ ] Set `REACT_APP_API_URL` in `.env` (§7, §8)

### Testing

- [ ] Smoke test each agent end-to-end with Snowflake data loaded
- [ ] Verify session persists across process restart (Snowflake store)
- [ ] Verify session history loads correctly in React
- [ ] Verify Bokeh chart renders in React with `embed_item`
- [ ] Verify file upload parses correctly and populates session state
- [ ] Verify confirmation flow (yes/no/edit) behaves correctly end-to-end
- [ ] Verify `/export` writes a valid Snowflake table
