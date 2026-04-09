#!/usr/bin/env python3
"""Lightweight end-to-end smoke test for the FinCast prototype."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_EXTERNAL_IMPORTS = [
    "dotenv",
    "groq",
    "numpy",
    "pandas",
    "plotly",
    "shap",
    "sklearn",
    "streamlit",
    "supabase",
]

REQUIRED_PROJECT_IMPORTS = [
    "analysis.data_connection",
    "analysis.recommendation_engine",
    "etl.transform",
    "models.explainability",
    "models.svr_pipeline",
]

REQUIRED_DIRS = [
    "analysis",
    "analysis/reports",
    "auth",
    "data",
    "etl",
    "models",
    "scripts",
]

EXPECTED_ARTIFACTS = [
    "analysis/reports/svr_evaluation_metrics.csv",
    "analysis/reports/svr_future_predictions.csv",
    "analysis/reports/phase_5_shap_global_importance.csv",
    "analysis/reports/phase_5_shap_local_explanations.csv",
    "analysis/reports/phase_5_summary.txt",
]

REQUIRED_ENV_KEYS = [
    "ALPHAVANTAGE_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "GROQ_API_KEY",
]


def try_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:  # pragma: no cover - smoke-test fallback
        return False, str(exc)


def load_env_from_dotenv(root: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return env_map

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_map[key.strip()] = value.strip().strip('"').strip("'")
    return env_map


def run_smoke_test(strict_env: bool, strict_artifacts: bool) -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    print("=" * 68)
    print("FinCast Prototype Smoke Test")
    print("=" * 68)

    failures: list[str] = []

    print("\n[1/4] Checking external dependencies...")
    for module in REQUIRED_EXTERNAL_IMPORTS:
        ok, detail = try_import(module)
        if ok:
            print(f"  [PASS] import {module}")
        else:
            print(f"  [FAIL] import {module}: {detail}")
            failures.append(f"Missing/broken external module: {module}")

    print("\n[2/4] Checking project module imports...")
    for module in REQUIRED_PROJECT_IMPORTS:
        ok, detail = try_import(module)
        if ok:
            print(f"  [PASS] import {module}")
        else:
            print(f"  [FAIL] import {module}: {detail}")
            failures.append(f"Broken project import: {module}")

    print("\n[3/4] Checking required folder structure...")
    for rel_path in REQUIRED_DIRS:
        path = root / rel_path
        if path.exists() and path.is_dir():
            print(f"  [PASS] dir {rel_path}")
        else:
            print(f"  [FAIL] missing dir {rel_path}")
            failures.append(f"Missing directory: {rel_path}")

    print("\n[4/4] Checking env keys and generated artifacts...")
    env_map = load_env_from_dotenv(root)
    missing_env = [k for k in REQUIRED_ENV_KEYS if not env_map.get(k)]
    if missing_env:
        level = "FAIL" if strict_env else "WARN"
        print(f"  [{level}] missing env keys in .env: {', '.join(missing_env)}")
        if strict_env:
            failures.append("Missing required .env keys")
    else:
        print("  [PASS] required env keys present in .env")

    present_count = 0
    for rel_path in EXPECTED_ARTIFACTS:
        if (root / rel_path).exists():
            present_count += 1

    if present_count == len(EXPECTED_ARTIFACTS):
        print(f"  [PASS] artifacts present: {present_count}/{len(EXPECTED_ARTIFACTS)}")
    else:
        level = "FAIL" if strict_artifacts else "WARN"
        print(f"  [{level}] artifacts present: {present_count}/{len(EXPECTED_ARTIFACTS)}")
        if strict_artifacts:
            failures.append("Missing expected report artifacts")

    print("\n" + "-" * 68)
    if failures:
        print("SMOKE TEST RESULT: FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("SMOKE TEST RESULT: PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinCast smoke tests")
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail if required .env keys are missing.",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Fail if expected report artifacts are missing.",
    )
    args = parser.parse_args()

    raise SystemExit(run_smoke_test(args.strict_env, args.strict_artifacts))


if __name__ == "__main__":
    main()
