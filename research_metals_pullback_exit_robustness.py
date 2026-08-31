#!/usr/bin/env python3
"""
METALS PULLBACK — EXIT ROBUSTNESS

We already learned:
- Pullback entries produced >60% winning trades.
- The strategy still lost slightly in Development because winners were too
  small relative to the stopped losses.

This test does NOT change:
- universe
- entry signal
- risk per trade
- stop
- fees/slippage
- max positions

It tests only how long we allow a normal mean-reversion winner to run:
exit above SMA 5, 8, or 10.

Selection rule:
- Candidate is selected ONLY from Development (2018 -> OOS boundary).
- OOS is diagnostic only; it is not used to choose the exit window.
- We require neighborhood robustness: at least 2 of 3 exit windows must
  have positive Development expectancy.
- Passing this test can only authorize fresh forward PAPER trading.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research_metals_stocks as base
import research_metals_pullback as pb


EXIT_WINDOWS = [5, 8, 10]

OUT_SUMMARY = Path("metals_pullback_exit_summary.json")
OUT_COMPARISON = Path("metals_pullback_exit_comparison.csv")
OUT_YEARLY = Path("metals_pullback_exit_yearly.csv")
OUT_TRADES = Path("metals_pullback_exit_trades.csv")


def make_prepared(data, exit_window):
    out = {}

    for sym in base.TRADE_SYMBOLS:
        if sym not in data:
            continue

        df = pb.prepare(data[sym], data[base.BENCHMARK]).copy()

        # Preserve the ORIGINAL entry condition: close < SMA5.
        df["entry_sma5"] = df["sma5"]

        # pb.simulate uses the "sma5" field for the exit test.
        # Replace only that exit reference with the requested MA.
        df["sma5"] = df["close"].rolling(exit_window).mean()

        df = df.dropna().copy()
        out[sym] = df

    return out


def fixed_entry_signal(row):
    return bool(
        bool(row["spy_regime"])
        and row["close"] > row["ema200"]
        and row["ema50"] > row["ema200"]
        and row["rsi2"] <= pb.RSI_ENTRY
        and row["close"] < row["entry_sma5"]
    )


def stats(trades, equity):
    return pb.metrics(trades, equity)


def yearly(exit_window, trades, period):
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(t) for t in trades])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    rows = []

    for y, g in df.groupby("year"):
        pnl = g["net_pnl"].astype(float)
        gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
        gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "exit_sma": exit_window,
            "period": period,
            "year": int(y),
            "trades": int(len(g)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def profitable_year_ratio(year_df):
    if year_df.empty:
        return 0.0, 0, 0

    evaluable = year_df[year_df["trades"] >= 8]

    if evaluable.empty:
        return 0.0, 0, 0

    positive = int((evaluable["net_pnl"] > 0).sum())
    total = int(len(evaluable))

    return positive / total, positive, total


def main():
    data = base.download_data()

    # Dates must be identical across all candidates.
    probe = make_prepared(data, 5)
    end = max(df.index.max() for df in probe.values())
    oos_start = (end - pd.DateOffset(months=base.OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(base.START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    original_entry = pb.entry_signal
    pb.entry_signal = fixed_entry_signal

    results = {}
    yearly_frames = []
    comparison_rows = []
    trade_rows = []

    try:
        for w in EXIT_WINDOWS:
            prepared = make_prepared(data, w)

            dev_trades, dev_eq = pb.simulate(
                f"EXIT_SMA{w}_Development",
                prepared,
                dev_start,
                dev_end,
            )

            oos_trades, oos_eq = pb.simulate(
                f"EXIT_SMA{w}_OOS",
                prepared,
                oos_start,
                end,
            )

            dev = stats(dev_trades, dev_eq)
            oos = stats(oos_trades, oos_eq)

            ydev = yearly(w, dev_trades, "Development")
            yoos = yearly(w, oos_trades, "OOS_diagnostic")
            ratio, pos_years, eval_years = profitable_year_ratio(ydev)

            dev_row = {
                "exit_sma": w,
                "period": "Development",
                **dev,
                "profitable_year_ratio": round(ratio, 3),
                "positive_years": pos_years,
                "evaluable_years": eval_years,
            }

            oos_row = {
                "exit_sma": w,
                "period": "OOS_diagnostic",
                **oos,
            }

            comparison_rows.extend([dev_row, oos_row])
            yearly_frames.extend([ydev, yoos])

            for validation_period, trades in [
                ("Development", dev_trades),
                ("OOS_diagnostic", oos_trades),
            ]:
                for t in trades:
                    row = asdict(t)
                    row["exit_sma"] = w
                    row["validation_period"] = validation_period
                    trade_rows.append(row)

            results[w] = {
                "dev": dev,
                "oos": oos,
                "profitable_year_ratio": ratio,
                "positive_years": pos_years,
                "evaluable_years": eval_years,
            }

    finally:
        pb.entry_signal = original_entry

    # Development only determines the candidate.
    positive_dev_expectancy_count = sum(
        1 for w in EXIT_WINDOWS if results[w]["dev"]["expectancy"] > 0
    )

    eligible_dev_candidates = []

    for w in EXIT_WINDOWS:
        d = results[w]["dev"]
        ratio = results[w]["profitable_year_ratio"]

        if (
            d["trades"] >= 300
            and d["net_pnl"] > 0
            and d["win_rate_pct"] >= 52.0
            and d["profit_factor"] >= 1.10
            and d["expectancy"] > 0
            and d["max_drawdown_pct"] <= 8.0
            and ratio >= 0.50
        ):
            eligible_dev_candidates.append(w)

    selected = None

    if eligible_dev_candidates and positive_dev_expectancy_count >= 2:
        # Choose by Development PF, then expectancy; never by OOS.
        selected = max(
            eligible_dev_candidates,
            key=lambda w: (
                results[w]["dev"]["profit_factor"],
                results[w]["dev"]["expectancy"],
            ),
        )

    oos_diagnostic_ok = False

    if selected is not None:
        o = results[selected]["oos"]

        oos_diagnostic_ok = bool(
            o["trades"] >= 60
            and o["net_pnl"] > 0
            and o["win_rate_pct"] >= 52.0
            and o["profit_factor"] >= 1.15
            and o["expectancy"] > 0
            and o["max_drawdown_pct"] <= 8.0
        )

    eligible_forward = bool(selected is not None and oos_diagnostic_ok)

    summary = {
        "test": "METALS_PULLBACK_EXIT_ROBUSTNESS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
        },
        "exit_windows_tested": EXIT_WINDOWS,
        "selection_uses_oos": False,
        "positive_development_expectancy_windows": positive_dev_expectancy_count,
        "results": {str(k): v for k, v in results.items()},
        "eligible_development_candidates": eligible_dev_candidates,
        "selected_exit_sma": selected,
        "oos_diagnostic_ok_for_selected": oos_diagnostic_ok,
        "eligible_for_fresh_forward_paper": eligible_forward,
        "decision": (
            "ELIGIBLE: freeze selected exit and start fresh forward paper"
            if eligible_forward
            else "FAIL: do not advance this pullback family"
        ),
        "real_money_status": "NOT APPROVED",
        "fresh_forward_requirements_if_eligible": {
            "minimum_calendar_months": 6,
            "minimum_closed_trades": 30,
            "profit_factor_min": 1.25,
            "win_rate_min_pct": 52.0,
            "expectancy_positive": True,
            "max_drawdown_pct": 10.0,
            "parameters_frozen": True,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(comparison_rows).to_csv(OUT_COMPARISON, index=False)

    pd.concat(
        [x for x in yearly_frames if not x.empty],
        ignore_index=True,
    ).to_csv(OUT_YEARLY, index=False)

    pd.DataFrame(trade_rows).to_csv(OUT_TRADES, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
