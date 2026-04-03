"""
Parameter schemas for each skill. Loaded from skills_config.yaml at startup
and used by the orchestrator to determine which parameters are missing and
how to ask for them.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParameterSpec:
    name: str
    label: str
    description: str
    data_type: str              # string | integer | choice | list | file | date
    required: bool = True
    choices: Optional[list] = None
    aliases: Optional[dict] = None   # choice -> list of synonyms
    default: Optional[Any] = None


class SkillSchema:
    def __init__(self, skill_id: str, config: dict):
        self.skill_id = skill_id
        self.display_name = config["display_name"]
        self.description = config["description"]
        self.required_parameters: list[ParameterSpec] = [
            ParameterSpec(**{**p, "required": True})
            for p in config.get("required_parameters", [])
        ]
        self.optional_parameters: list[ParameterSpec] = [
            ParameterSpec(**{**p, "required": False})
            for p in config.get("optional_parameters", [])
        ]

    def all_parameters(self) -> list[ParameterSpec]:
        return self.required_parameters + self.optional_parameters

    def get_missing_required(self, collected: dict) -> list[ParameterSpec]:
        """Return required parameters not yet present in collected dict."""
        return [
            p for p in self.required_parameters
            if collected.get(p.name) is None
        ]

    def normalize_choice(self, param_name: str, raw_value: str) -> Optional[str]:
        """Normalize a user-provided value to a canonical choice using aliases."""
        spec = next((p for p in self.all_parameters() if p.name == param_name), None)
        if spec is None or spec.choices is None:
            return raw_value
        raw_lower = raw_value.strip().lower()
        for choice in spec.choices:
            if raw_lower == choice.lower():
                return choice
            aliases = (spec.aliases or {}).get(choice, [])
            if raw_lower in [a.lower() for a in aliases]:
                return choice
        return None   # Not recognized


def load_schemas(config_path: str = None) -> dict[str, SkillSchema]:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "skills_config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return {
        skill_id: SkillSchema(skill_id, cfg)
        for skill_id, cfg in raw["skills"].items()
    }
