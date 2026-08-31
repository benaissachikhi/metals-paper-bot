#!/usr/bin/env python3
"""
Railway worker for the frozen Metals Rotation Forward Paper system.

PAPER ONLY.
- No broker connection.
- No API keys.
- No order placement.
- Reconstructs the frozen forward paper account from 2026-08-31.
- Writes snapshots for dashboard.py.
"""

import os
import time
import traceback
from pathlib import Path

import metals_rotation_forward_paper as fwd

REFRESH_SECONDS = int(os.getenv("ROTATION_REFRESH_SECONDS", "3600"))

preferred = Path(os.getenv("DATA_DIR", "/data"))
try:
    preferred.mkdir(parents=True, exist_ok=True)
    test = preferred / ".write_test"
    test.write_text("ok", encoding="utf-8")
    test.unlink(missing_ok=True)
    DATA_DIR = preferred
except Exception:
    DATA_DIR = Path(".").resolve()

# Redirect the frozen forward outputs to persistent Railway storage.
fwd.OUT_SUMMARY = DATA_DIR / "metals_rotation_forward_summary.json"
fwd.OUT_TRADES = DATA_DIR / "metals_rotation_forward_trades.csv"
fwd.OUT_EQUITY = DATA_DIR / "metals_rotation_forward_equity.csv"
fwd.OUT_POSITIONS = DATA_DIR / "metals_rotation_forward_positions.csv"
fwd.OUT_EXPOSURE = DATA_DIR / "metals_rotation_forward_exposure.csv"
fwd.OUT_REPORT = DATA_DIR / "metals_rotation_forward_report.md"


def update_snapshot():
    print("[rotation-paper] Updating frozen forward snapshot...", flush=True)
    fwd.main()
    print(
        f"[rotation-paper] Snapshot ready in {DATA_DIR}. "
        "PAPER ONLY — no real orders.",
        flush=True,
    )


def main():
    print(
        "[rotation-paper] Metals Rotation Forward Paper worker ONLINE. "
        "No broker connection. No real orders.",
        flush=True,
    )

    while True:
        try:
            update_snapshot()
        except Exception:
            print("[rotation-paper] Snapshot update failed:", flush=True)
            traceback.print_exc()

        time.sleep(max(900, REFRESH_SECONDS))


if __name__ == "__main__":
    main()
