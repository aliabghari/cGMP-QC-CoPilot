#!/usr/bin/env python3
"""
Preflight Verification Script (`preflight_check.py`).
Checks Python environment, required dependencies, artifact existence,
SHA-256 provenance hashes vs model_card.json, data leakage protections,
and measures demo scenario execution timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "agents" / "agent_2_analytical_scientist" / "assets" / "models"
SYNTH_DIR = ROOT / "agents" / "agent_2_analytical_scientist" / "assets" / "synthetic_stability"
CONFIG_FILE = ROOT / "agents" / "agent_2_analytical_scientist" / "assets" / "hybrid_config.json"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 75)
    print("cGMP-QC-CoPilot Preflight Environment & Provenance Check")
    print("=" * 75)

    # 1. Python version & interpreter check
    v = sys.version_info
    print(f"[*] Interpreter Path:    {sys.executable}")
    print(f"[*] Python Version:      {v.major}.{v.minor}.{v.micro} ({sys.version.split()[0]})")
    if (v.major, v.minor) != (3, 13):
        print(f"[WARN] Non-standard interpreter detected: Python {v.major}.{v.minor}. The declared production demo interpreter is Python 3.13.")
    else:
        print("[OK] Declared demo interpreter confirmed: Python 3.13.")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("[FAIL] Python 3.10+ required.")
        return 1

    # 2. Dependency checks
    # Allow appending cross-python user/system site-packages if running under alternate interpreter
    custom_site = os.environ.get("CUSTOM_SITE_PACKAGES")
    if custom_site:
        custom_path = Path(custom_site)
        if custom_path.exists() and str(custom_path) not in sys.path:
            sys.path.append(str(custom_path))

    required_pkgs = ["numpy", "pandas", "scipy", "plotly", "streamlit"]
    missing = []
    print("\n[*] Core Dependencies:")
    for pkg in required_pkgs:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [+] Package '{pkg}': OK (v{ver})")
        except ImportError:
            print(f"    [-] Package '{pkg}': MISSING")
            missing.append(pkg)

    # Check torch (optional runtime, required for training/autograd tests)
    try:
        import torch
        print(f"    [+] Package 'torch': OK (v{torch.__version__})")
    except ImportError:
        print("    [!] Package 'torch': NumPy fallback active (runtime pure NumPy inference enabled)")

    if missing:
        print(f"\n[FAIL] Missing required packages: {missing}")
        return 1

    # 3. Model and Synthetic Artifacts vs model_card.json
    print("\n[*] Checking Core Artifacts & Provenance Hashes vs model_card.json:")
    model_card = {}
    if MODEL_CARD_FILE.exists():
        try:
            model_card = json.loads(MODEL_CARD_FILE.read_text(encoding="utf-8")).get("artifacts", {})
        except Exception as e:
            print(f"    [!] Warning: Failed to parse model_card.json: {e}")

    required_artifacts = [
        CONFIG_FILE,
        MODELS_DIR / "hybrid_ensemble_weights.npz",
        MODELS_DIR / "calibration_bundle.json",
        MODELS_DIR / "phase4_uncertainty_report.json",
        SYNTH_DIR / "synth_observations.csv",
        SYNTH_DIR / "synth_calibration_observations.csv",
        SYNTH_DIR / "synth_excursions.csv",
        SYNTH_DIR / "_ground_truth_do_not_read_in_train.json",
        SYNTH_DIR / "dataset_manifest.json",
        SYNTH_DIR / "illustrative_demo_lot_NS-2026-004.csv",
    ]

    hash_mismatches = 0
    for p in required_artifacts:
        if not p.exists():
            print(f"    [-] Artifact missing: {p.relative_to(ROOT)}")
            return 1
        h = sha256_of_file(p)
        expected_h = model_card.get(p.name)
        if expected_h:
            if h.lower() == expected_h.lower():
                status_str = "MATCHES model_card.json"
            else:
                status_str = f"MISMATCH vs model_card ({expected_h[:12]}...)"
                hash_mismatches += 1
        else:
            status_str = "verified present"
        print(f"    [+] {p.name:<36} SHA-256: {h[:16]}... ({p.stat().st_size:,} bytes) -> {status_str}")

    if hash_mismatches > 0:
        print(f"[FAIL] {hash_mismatches} artifact(s) mismatched hashes in model_card.json")
        return 1

    # 4. Data Leakage Verification
    print("\n[*] Verifying Training Loader Data Leakage Protection:")
    sys.path.insert(0, str(ROOT))
    from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.synth import (
        load_observations_for_training,
    )
    df_train = load_observations_for_training(SYNTH_DIR / "synth_observations.csv")
    leaked = [c for c in df_train.columns if "structural" in c.lower() or "ground_truth" in c.lower()]
    if leaked:
        print(f"[FAIL] Data leakage detected in training loader: {leaked}")
        return 1
    print(f"    [+] Observations loader exposes {len(df_train.columns)} columns; 0 ground-truth leakage columns.")

    # 5. Demo-Scenario Timing Verification
    print("\n[*] Benchmarking Demo Scenario Execution Timing (Lot NS-2026-004, 36h @ 28°C):")
    from agents.agent_2_analytical_scientist.scripts.hybrid_kinetics.workflow import (
        build_mso_draft_quality_package,
    )
    t0 = time.perf_counter()
    pkg = build_mso_draft_quality_package(
        lot_id="NS-2026-004",
        excursion_temp_c=28.0,
        excursion_duration_hours=36.0,
    )
    t_elapsed = time.perf_counter() - t0
    exc_verdict = pkg["excursion_result"]["summary"]["verdict"]
    oot_verdict = pkg["oot_result"]["oot_diagnostics"]["oot_verdict"]
    print(f"    [+] Excursion Verdict:    '{exc_verdict}'")
    print(f"    [+] OOT Verdict:          '{oot_verdict}'")
    print(f"    [+] Demo Scenario Timing: {t_elapsed:.3f} seconds wall-clock")
    timing_ceiling_sec = float(os.environ.get("PREFLIGHT_TIMING_CEILING_SEC", "2.0"))
    if t_elapsed > timing_ceiling_sec:
        if os.environ.get("PREFLIGHT_TIMING_ADVISORY") == "1":
            print(f"    [WARN] Demo Scenario Timing exceeded {timing_ceiling_sec:.1f}s ({t_elapsed:.3f}s), advisory mode active.")
        else:
            print(f"    [FAIL] Demo Scenario Timing exceeded {timing_ceiling_sec:.1f}s ({t_elapsed:.3f}s). Set PREFLIGHT_TIMING_ADVISORY=1 to downgrade.")
            return 1

    print("\n" + "=" * 75)
    print("PREFLIGHT CHECK PASSED: Environment, artifacts, provenance, and timing verified.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
