"""Project configuration loader."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the global configuration YAML file."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def paths(config: dict) -> dict:
    """Resolve absolute data/output paths from config."""
    d = config["data"]
    return {
        "raw_public": PROJECT_ROOT / d["raw_public"],
        "processed": PROJECT_ROOT / d["processed"],
        "results": PROJECT_ROOT / d["results"],
        "tables": PROJECT_ROOT / "output" / "tables",
        "figures": PROJECT_ROOT / "output" / "figures",
    }
