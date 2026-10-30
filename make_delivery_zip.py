"""
make_delivery_zip.py
Automated Delivery Packaging, Linter, Test, Real Git Round-Trip & Streamlit Smoke Test Verification for cGMP-QC-CoPilot.

Packaging Strategy:
- Implements explicit deny-list matching .gitignore, with an explicit allow-list for tracked
  results/*.json files required by test suites.
- Strictly excludes cache, virtual environments, VCS, local databases, scratch scripts,
  historical AI showcase files, and results/screenshots/.
"""

import os
import re
import sys
import json
import shutil
import hashlib
import zipfile
import tempfile
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = ROOT_DIR / "cGMP-QC-CoPilot-Delivery.zip"

DENY_DIR_NAMES = {
    ".git",
    ".git-backup",
    ".venv",
    "venv",
    "ENV",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "build",
    "dist",
    "synth_schedule",
}

DENY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".sqlite3",
    ".zip",
    ".log",
    ".swp",
    ".swo",
}

DENY_EXACT_FILES = {
    "compute_float_parity.py",
    "compute_hist_comparators.py",
    "recompute_15c_audit.py",
    "recompute_shelf_life_truth.py",
    "run_grep_audit.py",
    "verify_hybrid.py",
    "ANTIGRAVITY_PROJECT_SHOWCASE.md",
    "VERIFICATION_REPORT.md",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

TRACKED_RESULTS_JSON = {
    Path("results/final_metrics.json"),
    Path("results/wp1_metrics.json"),
    Path("results/spc_performance.json"),
    Path("results/wp2_edit_proof.json"),
    Path("results/render_times.json"),
}


def should_exclude(rel_path: Path) -> bool:
    # 1. Output zip self-exclusion
    if rel_path.name == OUTPUT_ZIP.name:
        return True

    # 2. Explicit Allow-list: Tracked results/*.json must be shipped
    if rel_path in TRACKED_RESULTS_JSON:
        return False

    # 3. Deny results/screenshots/
    if len(rel_path.parts) >= 2 and rel_path.parts[0] == "results" and rel_path.parts[1] == "screenshots":
        return True

    # 4. Deny exact files
    if rel_path.name in DENY_EXACT_FILES:
        return True

    # 5. Deny extensions
    if rel_path.suffix.lower() in DENY_EXTENSIONS:
        return True

    # 6. Deny directory names
    for part in rel_path.parts[:-1]:
        if part in DENY_DIR_NAMES:
            return True

    return False


def build_delivery_zip() -> tuple[int, float]:
    print(f"[1/4] Packaging project into {OUTPUT_ZIP.name}...")
    file_count = 0

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in DENY_DIR_NAMES and d != "screenshots"]

            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(ROOT_DIR)

                if should_exclude(rel_path):
                    continue

                zf.write(abs_path, arcname=str(rel_path).replace("\\", "/"))
                file_count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"  [OK] Packaged {file_count} files ({size_mb:.2f} MB).")
    return file_count, size_mb


def self_verify_delivery_zip(file_count: int, size_mb: float):
    print("[2/4] Self-verifying delivery package in isolated extraction directory...")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # 1. Unzip
        with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
            zf.extractall(tmp_dir)

        # 2. Strict content assertions on extracted directory
        for r_file in TRACKED_RESULTS_JSON:
            assert (tmp_dir / r_file).exists(), f"Missing required tracked results file: {r_file}"

        for p in tmp_dir.rglob("*"):
            rel = p.relative_to(tmp_dir)
            assert p.name != "synth_schedule", f"Banned folder found: {rel}"
            assert p.name not in DENY_EXACT_FILES, f"Banned file found: {rel}"
            assert not (p.is_file() and p.suffix.lower() in {".db", ".zip"}), f"Banned file type: {rel}"
            assert not (len(rel.parts) >= 2 and rel.parts[0] == "results" and rel.parts[1] == "screenshots"), f"screenshots found: {rel}"
            for part in rel.parts:
                assert part not in {".ruff_cache", "__pycache__", ".pytest_cache"}, f"Cache folder found: {rel}"

        # Leak check for user paths
        pattern = re.compile(r"Users[\\/]Ali|AppData", re.IGNORECASE)
        path_leaks = []
        for p in tmp_dir.rglob("*"):
            if p.is_file() and p.name != "make_delivery_zip.py" and p.suffix.lower() in {".py", ".json", ".md", ".csv", ".txt", ".yml", ".yaml", ".toml"}:
                try:
                    txt = p.read_text(encoding="utf-8")
                    if pattern.search(txt):
                        path_leaks.append(str(p.relative_to(tmp_dir)))
                except Exception:
                    continue
        assert not path_leaks, f"User path leaks found: {path_leaks}"
        print("  [OK] Zip extraction assertions passed (no leaks, no banned files, results/*.json intact).")


def git_round_trip_verification():
    print("[3/4] Performing real git round-trip verification in temp directory...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        zip_files = set(zf.namelist())

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        repo_dir = tmp / "repo"
        repo_dir.mkdir()

        # 1. Copy working tree excluding .git, .venv, caches, *.zip
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in DENY_DIR_NAMES and d != "screenshots"]
            for f in files:
                rel = (Path(root) / f).relative_to(ROOT_DIR)
                if not should_exclude(rel):
                    dest = repo_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(root) / f, dest)

        # 2. git init, config, add, commit
        subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "verify"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "verify@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

        res_add = subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True, text=True)
        res_commit = subprocess.run(["git", "commit", "-m", "verify"], cwd=str(repo_dir), capture_output=True, text=True)

        add_commit_combined = (res_add.stdout or "") + "\n" + (res_add.stderr or "") + "\n" + (res_commit.stdout or "") + "\n" + (res_commit.stderr or "")

        # 3. Fail if git printed line-ending replacement warnings for agents/**/assets
        asset_warning_re = re.compile(r"(CRLF will be replaced by LF|LF will be replaced by CRLF).*agents[\\/].*[\\/]assets", re.IGNORECASE)
        for line in add_commit_combined.splitlines():
            if asset_warning_re.search(line):
                raise RuntimeError(f"Prohibited line ending warning for asset file: {line}")
        print("  [OK] Git add/commit completed with zero line-ending warnings on agents/**/assets.")

        # 4. git clone
        clone_dir = tmp / "clone"
        subprocess.run(["git", "clone", str(repo_dir), str(clone_dir)], check=True, capture_output=True)
        print("  [OK] Git clone succeeded.")

        # 5. Assert clone file list equals zip file list (excluding .git)
        clone_files = set()
        for root, dirs, files in os.walk(clone_dir):
            dirs[:] = [d for d in dirs if d != ".git"]
            for f in files:
                rel = (Path(root) / f).relative_to(clone_dir)
                clone_files.add(str(rel).replace("\\", "/"))

        diff_zip_clone = zip_files.symmetric_difference(clone_files)
        assert not diff_zip_clone, f"Zip and clone file lists differ: {diff_zip_clone}"
        print(f"  [OK] Clone file list exactly matches zip file list ({len(clone_files)} files).")

        # 6. Assert in clone: SHA-256 of every file listed in model_card.json equals recorded value
        model_card_path = clone_dir / "agents" / "agent_2_analytical_scientist" / "assets" / "models" / "model_card.json"
        with open(model_card_path, "r", encoding="utf-8") as mc_f:
            mc = json.load(mc_f)

        assets_base = clone_dir / "agents" / "agent_2_analytical_scientist" / "assets"
        for asset in mc.get("benchmarking_artifacts", []):
            rel_p = asset.get("relative_path")
            expected_hash = asset.get("sha256")
            actual_p = assets_base / rel_p
            assert actual_p.exists(), f"Asset missing in clone: {rel_p}"
            actual_hash = hashlib.sha256(actual_p.read_bytes()).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch in clone for {rel_p}: expected {expected_hash}, got {actual_hash}"
        print("  [OK] All model_card.json SHA-256 provenance hashes verified in clone.")

        # Assert no user leaks or banned caches in clone
        leak_pat = re.compile(r"Users[\\/]Ali|AppData", re.IGNORECASE)
        for p in clone_dir.rglob("*"):
            if p.is_file() and ".git" not in p.parts and p.name != "make_delivery_zip.py" and p.suffix.lower() in {".py", ".json", ".md", ".csv", ".txt", ".yml", ".yaml", ".toml"}:
                try:
                    txt = p.read_text(encoding="utf-8")
                    if leak_pat.search(txt):
                        raise AssertionError(f"User path leak in clone: {p.relative_to(clone_dir)}")
                except Exception as e:
                    if isinstance(e, AssertionError):
                        raise
            if ".git" not in p.parts:
                assert not (p.is_file() and p.suffix.lower() in {".db", ".zip"}), f"Banned file type in clone: {p}"
                for part in p.parts:
                    assert part not in {".ruff_cache", "__pycache__", ".pytest_cache"}, f"Cache in clone: {p}"
        print("  [OK] Clone assertions passed (no leaks, no banned files/caches).")

        # 7. In clone: fresh venv, pip install requirements-dev.txt, ruff, preflight, pytest
        print("  [+] Building fresh virtual environment in clone...")
        venv_dir = clone_dir / ".venv"
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], check=True, capture_output=True)
        venv_py = venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"

        subprocess.run([str(venv_py), "-m", "pip", "install", "-r", "requirements-dev.txt"], cwd=str(clone_dir), check=True, capture_output=True)

        res_ruff = subprocess.run([str(venv_py), "-m", "ruff", "check", "."], cwd=str(clone_dir), capture_output=True, text=True)
        assert res_ruff.returncode == 0, f"Ruff failed in clone:\n{res_ruff.stderr}"
        print("  [OK] ruff check . passed in clone.")

        env_vars = os.environ.copy()
        env_vars["PREFLIGHT_TIMING_ADVISORY"] = "1"
        res_preflight = subprocess.run([str(venv_py), "preflight_check.py"], cwd=str(clone_dir), capture_output=True, text=True, env=env_vars)
        assert res_preflight.returncode == 0, f"Preflight failed in clone:\n{res_preflight.stderr}"
        for line in res_preflight.stdout.splitlines():
            if "Demo Scenario Timing:" in line:
                print(f"  [OK] {line.strip()}")

        res_pytest = subprocess.run([str(venv_py), "-m", "pytest", "-q"], cwd=str(clone_dir), capture_output=True, text=True)
        assert res_pytest.returncode == 0, f"Pytest failed in clone:\n{res_pytest.stderr}"
        last_line = res_pytest.stdout.strip().splitlines()[-1] if res_pytest.stdout.strip() else ""
        print(f"  [OK] pytest passed in clone: {last_line}")

        # 8. Streamlit smoke test on clone (FIX 4)
        print("[4/4] Executing Streamlit smoke test against CLONE on port 8599...")
        st_proc = subprocess.Popen(
            [str(venv_py), "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8599", "--browser.gatherUsageStats", "false"],
            cwd=str(clone_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        st_ok = False
        for i in range(30):
            time.sleep(1)
            for url in ["http://localhost:8599/_stcore/health", "http://127.0.0.1:8599/_stcore/health"]:
                try:
                    with urllib.request.urlopen(url, timeout=1) as resp:
                        body = resp.read().decode("utf-8").strip()
                        if body == "ok":
                            st_ok = True
                            break
                except Exception:
                    pass
            if st_ok:
                break

        st_proc.terminate()
        try:
            st_proc.wait(timeout=5)
        except Exception:
            st_proc.kill()

        assert st_ok, "Streamlit health check on clone failed to return 'ok'"
        print("  [OK] Streamlit health check on CLONE: PASS (http://localhost:8599/_stcore/health returned 'ok').")

    print("\n" + "=" * 60)
    print(f"DELIVERY PACKAGE READY & VERIFIED: {OUTPUT_ZIP.name}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("cGMP-QC-CoPilot Delivery Packaging & Verification")
    print("=" * 60)
    file_count, size_mb = build_delivery_zip()
    self_verify_delivery_zip(file_count, size_mb)
    git_round_trip_verification()


if __name__ == "__main__":
    main()
