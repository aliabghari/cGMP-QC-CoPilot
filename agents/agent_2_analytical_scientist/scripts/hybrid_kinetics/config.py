"""
Configuration loader, parameter bounds, and deterministic SHA-256 hashing for Hybrid Kinetics.
All parameter bounds are loaded from `agents/agent_2_analytical_scientist/assets/hybrid_config.json`
and marked "ASSUMPTION - SME to confirm".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "hybrid_config.json"
)


def load_hybrid_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads the hybrid kinetics configuration JSON."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config_sha256(config_path: str | Path | None = None) -> str:
    """Computes the SHA-256 hex digest of the configuration file for ALCOA+ audit logging."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw_bytes = path.read_bytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def get_parameter_bounds(
    param_name: str, config: Dict[str, Any] | None = None
) -> Tuple[float, float, float]:
    """
    Returns (min_val, max_val, nominal_val) for a named kinetic parameter from config.
    """
    if config is None:
        config = load_hybrid_config()
    p_meta = config["parameter_bounds"][param_name]
    return float(p_meta["min"]), float(p_meta["max"]), float(p_meta["nominal"])


def get_nominal_parameters(config: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Returns a dictionary of nominal parameter values from `hybrid_config.json`."""
    if config is None:
        config = load_hybrid_config()
    return {
        k: float(v["nominal"])
        for k, v in config["parameter_bounds"].items()
    }
