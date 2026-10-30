"""
CLI Entrypoint for Offline Hybrid Kinetic Ensemble Training (`scripts/train_hybrid.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.train import (
    train_hybrid_ensemble,
)


def main() -> None:
    summary = train_hybrid_ensemble()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
