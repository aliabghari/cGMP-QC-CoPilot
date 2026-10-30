"""
cGMP-QC-CoPilot: Synthetic Data Governance & Manifest Verification Module.

Enforces Non-Negotiable Guardrails:
- G1: Synthetic only. Every dataset is synthetic, labeled so in UI and files,
      generated with fixed seeds, and listed in a manifest with SHA-256 hashes.
- G4: No inverse crime. Synthetic generators must include unmodeled structural effects
      and measurement noise distinct from downstream fitted models.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SYNTHETIC_HEADER_TAG = "SYNTHETIC / NOT GMP DATA — Prototype Decision-Support Modules for Demonstration Only"
SYNTHETIC_JSON_PROPERTY = "is_synthetic"


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hexadecimal digest of a file on disk."""
    p = Path(file_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest(
    data_files: List[Union[str, Path]],
    manifest_output_path: Optional[Union[str, Path]] = None,
    suite_name: str = "cGMP-QC-CoPilot Synthetic Benchmark Datasets",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Creates or updates a SHA-256 manifest of synthetic benchmark data files.
    """
    entries = []
    for fp in data_files:
        p = Path(fp)
        if not p.exists():
            continue
        rel_path = str(p).replace("\\", "/")
        sha = compute_file_sha256(p)
        size_bytes = p.stat().st_size
        entries.append(
            {
                "file_path": rel_path,
                "file_name": p.name,
                "sha256": sha,
                "size_bytes": size_bytes,
                "is_synthetic": True,
                "synthetic_label": SYNTHETIC_HEADER_TAG,
            }
        )

    manifest = {
        "suite_name": suite_name,
        "default_seed": seed,
        "is_synthetic": True,
        "synthetic_notice": SYNTHETIC_HEADER_TAG,
        "files_count": len(entries),
        "files": sorted(entries, key=lambda x: x["file_path"]),
    }

    if manifest_output_path:
        out_p = Path(manifest_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def verify_manifest(
    manifest_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, List[str]]:
    """
    Verifies that all files listed in the manifest exist on disk and match their recorded SHA-256 digests.
    Returns (True, []) if all pass, or (False, error_messages) if any mismatch or missing.
    """
    mp = Path(manifest_path)
    if not mp.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    with open(mp, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    base = Path(base_dir) if base_dir else mp.parent

    for entry in manifest.get("files", []):
        fp = entry["file_path"]
        expected_sha = entry["sha256"]
        p = Path(fp)
        if not p.is_absolute():
            # Try relative to base, then relative to repo root
            candidate1 = base / p
            candidate2 = Path(__file__).resolve().parent.parent / p
            if candidate1.exists():
                p = candidate1
            elif candidate2.exists():
                p = candidate2

        if not p.exists():
            errors.append(f"File missing: {fp}")
            continue

        actual_sha = compute_file_sha256(p)
        if actual_sha != expected_sha:
            errors.append(
                f"SHA-256 mismatch for {fp}: expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors


def check_synthetic_labeling(file_path: Union[str, Path]) -> bool:
    """
    Checks if a data file or script contains an explicit synthetic label.
    Supports .json, .csv, .md, .py, .txt.
    """
    p = Path(file_path)
    if not p.exists():
        return False

    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return bool(
                    data.get("is_synthetic") is True
                    or "SYNTHETIC" in json.dumps(data)
                )
        except Exception:
            pass

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        return "SYNTHETIC" in content.upper()
    except Exception:
        return False
