"""
Thin wrapper around the Dataiku LLM Mesh API.
Replace YOUR_LLM_CONNECTION_ID in llm_config.yaml with the actual connection ID
configured in your Dataiku instance's LLM Mesh settings.
"""
import json
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: dict):
        self.connection_id = config["llm_mesh"]["connection_id"]
        self.max_tokens = config["llm_mesh"].get("max_tokens", 4096)
        self.temp_classify = config["llm_mesh"].get("temperature_classify", 0.1)
        self.temp_extract = config["llm_mesh"].get("temperature_extract", 0.1)
        self.temp_agents = config["llm_mesh"].get("temperature_agents", 0.3)
        self.temp_deterministic = config["llm_mesh"].get("temperature_deterministic", 0.0)
        self._client = None
        self.call_log: list[dict] = []

    def _get_dataiku_client(self):
        """Lazy-initialize the Dataiku API client."""
        if self._client is None:
            import dataiku
            self._client = dataiku.api_client()
        return self._client

    def complete(self, messages: list[dict], temperature: float = None) -> str:
        """
        Send a list of {role, content} messages to the LLM Mesh and return the
        text response. Temperature is managed by the Dataiku LLM Mesh connection
        configuration; Azure OpenAI via LLM Mesh does not support per-call
        temperature overrides through the Python SDK.
        """
        try:
            api_client = self._get_dataiku_client()
            project_key = __import__("dataiku").default_project_key()
            project = api_client.get_project(project_key)
            llm = project.get_llm(self.connection_id)

            completion = llm.new_completion()
            for msg in messages:
                completion.with_message(msg["content"], msg["role"])
            try:
                completion.with_max_output_tokens(self.max_tokens)
            except Exception:
                pass  # older Dataiku SDK versions may not support this

            resp = completion.execute()
            self.call_log.append({"messages": messages, "response": resp.text})
            return resp.text

        except Exception as e:
            logger.error("LLM Mesh call failed: %s", e)
            self.call_log.append({"messages": messages, "response": f"ERROR: {e}", "error": True})
            raise

    def complete_json(self, messages: list[dict], temperature: float = None) -> dict:
        """
        Like complete(), but expects the model to return a JSON object.
        Strips markdown code fences if present before parsing.
        """
        raw = self.complete(messages, temperature=temperature)
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> dict:
        text = raw.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to repair truncated JSON (e.g. response cut off by token limit)
            repaired = self._repair_json(text)
            try:
                result = json.loads(repaired)
                logger.warning("Parsed repaired (truncated) JSON — some fields may be incomplete")
                return result
            except json.JSONDecodeError as e:
                logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", e, raw)
                raise ValueError(f"LLM returned non-JSON output: {raw[:200]}")

    @staticmethod
    def _repair_json(text: str) -> str:
        """Close unclosed strings, arrays, and objects to make truncated JSON parseable."""
        # Walk the string tracking open/close state
        in_string = False
        escaped = False
        stack = []
        last_complete = 0

        for i, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
                last_complete = i
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
                last_complete = i

        # If we're mid-string, close it first
        if in_string:
            text += '"'
        # Close open arrays then objects
        closers = {"{": "}", "[": "]"}
        for opener in reversed(stack):
            text += closers[opener]
        return text
