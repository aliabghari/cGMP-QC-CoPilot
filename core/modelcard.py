"""
cGMP-QC-CoPilot: Model Card Schema and Management Engine.

Enforces Guardrail G5 (Honest Evaluation) and G2 (Non-GMP decision support):
Every predictive model, hybrid solver, or analytical engine must register a model card
with:
- model_id & model_name
- intended_use & out_of_scope
- training_data_summary (synthetic data provenance, seed, sample sizes)
- deterministic_baseline (comparison baseline specification)
- benchmark_results (held-out performance, confidence intervals, error reduction)
- known_failure_modes & mixed/negative results disclosure
- regulatory non-GMP disclaimer
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_DISCLAIMER = "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"


@dataclass
class ModelCard:
    model_id: str
    model_name: str
    version: str
    intended_use: str
    out_of_scope: List[str]
    training_data_summary: Dict[str, Any]
    deterministic_baseline: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    known_failure_modes: List[str]
    date: str = "2026-10-26"
    regulatory_disclaimer: str = DEFAULT_DISCLAIMER
    negative_or_mixed_results: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_model_card(card_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a model card dictionary against the mandatory cGMP-QC-CoPilot schema.
    Returns (True, []) or (False, list_of_errors).
    """
    errors: List[str] = []

    required_fields = [
        "model_id",
        "model_name",
        "version",
        "intended_use",
        "out_of_scope",
        "training_data_summary",
        "deterministic_baseline",
        "benchmark_results",
        "known_failure_modes",
    ]

    for req in required_fields:
        if req not in card_dict or card_dict[req] is None:
            errors.append(f"Missing required model card field: '{req}'")

    # Check non-empty failure modes (G5 honest disclosure)
    failure_modes = card_dict.get("known_failure_modes")
    if not isinstance(failure_modes, list) or len(failure_modes) == 0:
        errors.append("Model card must specify at least one known failure mode or limitation.")

    # Check baseline comparison
    baseline = card_dict.get("deterministic_baseline")
    if not isinstance(baseline, dict) or not baseline.get("baseline_name"):
        errors.append("Model card must specify deterministic_baseline with 'baseline_name'.")

    # Check benchmark results
    bench = card_dict.get("benchmark_results")
    if not isinstance(bench, dict) or len(bench) == 0:
        errors.append("Model card must include benchmark_results evaluated on held-out data.")

    # Check regulatory disclaimer
    disclaimer = card_dict.get("regulatory_disclaimer", "")
    if DEFAULT_DISCLAIMER not in str(disclaimer):
        errors.append(
            f"Model card must contain mandatory regulatory disclaimer: '{DEFAULT_DISCLAIMER}'"
        )

    return len(errors) == 0, errors


def save_model_card(card: Union[ModelCard, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Saves a model card to JSON, ensuring schema validity before writing."""
    if isinstance(card, ModelCard):
        card_data = card.to_dict()
    else:
        card_data = card

    valid, errors = validate_model_card(card_data)
    if not valid:
        raise ValueError(f"Cannot save invalid model card: {errors}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(card_data, f, indent=2)


def load_model_card(path: Union[str, Path]) -> ModelCard:
    """Loads and validates a ModelCard from JSON file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, errors = validate_model_card(data)
    if not valid:
        raise ValueError(f"Model card file {p} fails schema validation: {errors}")

    return ModelCard(
        model_id=data["model_id"],
        model_name=data["model_name"],
        version=data["version"],
        intended_use=data["intended_use"],
        out_of_scope=data["out_of_scope"],
        training_data_summary=data["training_data_summary"],
        deterministic_baseline=data["deterministic_baseline"],
        benchmark_results=data["benchmark_results"],
        known_failure_modes=data["known_failure_modes"],
        date=data.get("date", "2026-10-26"),
        regulatory_disclaimer=data.get("regulatory_disclaimer", DEFAULT_DISCLAIMER),
        negative_or_mixed_results=data.get("negative_or_mixed_results"),
        artifacts=data.get("artifacts", {}),
    )
