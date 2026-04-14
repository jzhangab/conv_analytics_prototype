"""
Notebook-specific patches applied on top of the standard backend.

Call ``apply_patches(orchestrator)`` from the Dataiku notebook to:

* Reload skill schemas from the project root (bypasses Dataiku workdir cache)

All other functionality (call logging, general-knowledge fallback,
confidence thresholds, shortened prompts) is now part of the standard backend.
"""
from __future__ import annotations

import os
from pathlib import Path

import backend
from backend.state.parameter_schema import load_schemas


def apply_patches(orchestrator):
    """Apply notebook-specific patches to *orchestrator*."""

    # Reload schemas from project root (bypasses Dataiku workdir cache)
    skills_cfg = Path(os.getcwd()) / "config" / "skills_config.yaml"
    if not skills_cfg.exists():
        skills_cfg = Path(backend.__file__).parent.parent / "config" / "skills_config.yaml"
    orchestrator.schemas = load_schemas(str(skills_cfg))

    # Summary
    print(f"Router skills:  {list(orchestrator.router._registry.keys())}")
    ws = orchestrator.web_search
    print(f"Web search:     {'enabled' if ws.enabled else 'disabled'}")
    print(f"Schemas loaded: {list(orchestrator.schemas.keys())}")
    print(f"LLM connection: {orchestrator.llm.connection_id}")
    print("All notebook patches applied.")
