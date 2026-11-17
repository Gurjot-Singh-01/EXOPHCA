#!/usr/bin/env python
"""One command that verifies the project: pip check, compileall, the test suite.

Prints a PASS/FAIL line per step and a summary, and exits non-zero if any step
fails. Standard library only, so it runs before (and independently of) an
editable install.

    python scripts/check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = (
    ("pip check", ("-m", "pip", "check")),
    (
        "compileall",
        ("-m", "compileall", "-q", "exophca", "app", "scripts"),
    ),
    ("tests", ("-m", "unittest", "discover", "-s", "tests", "-v")),
)


def run(label: str, arguments: tuple[str, ...]) -> bool:
    command = [sys.executable, *arguments]
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    code = subprocess.run(command, cwd=ROOT).returncode
    print(f"--- {label}: {'PASS' if code == 0 else 'FAIL'} (exit {code})", flush=True)
    return code == 0


def main() -> int:
    results = [(label, run(label, arguments)) for label, arguments in STEPS]

    print("\n=== summary ===")
    for label, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {label}")

    failed = [label for label, passed in results if not passed]
    if failed:
        print(f"\nFAIL: {', '.join(failed)}")
        return 1

    print("\nPASS: every check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
