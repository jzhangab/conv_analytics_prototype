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

    def _get_dataiku_client(self):
        """Lazy-initialize the Dataiku API client."""
        if self._client is None:
            import dataiku
            self._client = dataiku.api_client()
        return self._client

    def complete(self, messages: list[dict], temperature: float = None) -> str:
        """
        Send a list of {role, content} messages to the LLM Mesh and return the
        text response. temperature defaults to self.temp_agents if not specified.
        """
        if temperature is None:
            temperature = self.temp_agents

        try:
            api_client = self._get_dataiku_client()
            project_key = __import__("dataiku").default_project_key()
            project = api_client.get_project(project_key)
            llm = project.get_llm(self.connection_id)

            completion = llm.new_completion()
            for msg in messages:
                completion.with_message(msg["role"], msg["content"])

            resp = completion.execute(
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            return resp.text

        except Exception as e:
            logger.error("LLM Mesh call failed: %s", e)
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
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", e, raw)
            raise ValueError(f"LLM returned non-JSON output: {raw[:200]}")
