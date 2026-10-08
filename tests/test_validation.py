"""Test-runner adapter: the validation scripts ARE the test suite.

Every check in this project lives in ``scripts/validate_*.py``, each built to be
falsified first. This file only exposes them to standard runners, one test per
script, so CI picks them up without a second copy of any assertion:

    .venv/Scripts/python.exe -m unittest discover -s tests -v
    pytest tests/                                   # if pytest is installed

For a human-readable table, use ``scripts/run_all.py`` instead.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import neurocast  # noqa: E402
from run_all import discover  # noqa: E402


class ImportEverything(unittest.TestCase):
    """Every module imports. Catches a broken import before a 3-minute script does."""

    def test_all_modules_import(self) -> None:
        names = [m.name for m in pkgutil.walk_packages(neurocast.__path__, "neurocast.")]
        self.assertGreater(len(names), 30)
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(name)


class Validators(unittest.TestCase):
    """One test per ``scripts/validate_*.py``; each must exit 0."""


def _make(path: Path):
    def test(self: unittest.TestCase) -> None:
        proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
        self.assertEqual(proc.returncode, 0, f"{path.name} failed:\n{tail}")

    test.__doc__ = f"{path.name} exits 0"
    return test


for _p in discover():
    setattr(Validators, f"test_{_p.stem}", _make(_p))


if __name__ == "__main__":
    unittest.main()
