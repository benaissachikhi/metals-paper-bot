#!/usr/bin/env python3
"""
METALS ROTATION — FINAL 6% VOLATILITY VALIDATION

FINAL PRE-FORWARD VALIDATION.

The strategy family is already chosen:
- Monthly/21-session relative-strength rotation.
- Top 4.
- SPY regime filter.
- Same universe, ranking, fees and slippage.

The previous fixed 10% volatility overlay preserved the edge but its
Development max drawdown remained above the pre-registered 12% ceiling.

FINAL DECISION MADE BEFORE THIS TEST:
- Use a 6% annualized portfolio-volatility target.
- Do NOT search other volatility targets.
- No leverage.
- Unused capital remains cash.

Why 6%:
The 10% overlay had Development max DD ~17.8%. Scaling gross risk from
10% to 6% should materially reduce drawdown while preserving the same
signal edge, ranking and trade direction. This is a risk decision, not
a signal optimization.

PASS TO FRESH FORWARD PAPER requires ALL:
Development:
- >= 200 completed position-periods
- net return > 0
- win rate >= 52%
- PF >= 1.20
- expectancy > 0
- max DD <= 12%
- profitable-year ratio >= 62.5%
- >= 8 symbols with non-negative PnL
- positive PnL concentration <= 35%

OOS diagnostic:
- >= 50 completed position-periods
- net return > 0
- win rate >= 52%
- PF >= 1.20
- expectancy > 0
- max DD <= 10%

Passing authorizes ONLY fresh forward PAPER trading.
Real money remains NOT APPROVED.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import research_metals_rotation as rot
import research_metals_rotation_risk as risk


FINAL_TARGET_VOL = 0.06

OUT_SUMMARY = Path("metals_rotation_final6_summary.json")
OUT_COMPARISON = Path("metals_rotation_final6_comparison.csv")
OUT_YEARLY = Path("metals_rotation_final6_yearly.csv")
OUT_TRADES = Path("metals_rotation_final6_trades.csv")
OUT_EQUITY = Path("metals_rotation_final6_oos_equity.csv")
OUT_EXPOSURE = Path("metals_rotation_final6_exposure.csv")


def evaluate_target(features, start_dt, end_dt, label, target_vol):
    old = risk.TARGET_VOL
    risk.TARGET_VOL = target_vol
    try:
        trades, eq, exposure = risk.simulate_risk_control(
            label, features, start_dt, end_dt
        )
    finally:
        risk.TARGET_VOL = old

    return trades, eq, exposure


def main():
    data = rot.download_data()
    features = rot.build_features(data)

    end = max(df.index.max() for df in features.values())
    oos_start = (end - pd.DateOffset(months=rot.OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(rot.START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    # 10% is comparison only; 6% was selected BEFORE this validation.
    dev10_t, dev10_e, _ = evaluate_target(
        features, dev_start, dev_end, "VOL10_Development", 0.10
    )
    oos10_t, oos10_e, _ = evaluate_target(
        features, oos_start, end, "VOL10_OOS", 0.10
    )

    dev6_t, dev6_e, dev6_x = evaluate_target(
        features, dev_start, dev_end, "VOL6_Development", FINAL_TARGET_VOL
    )
    oos6_t, oos6_e, oos6_x = evaluate_target(
        features, oos_start, end, "VOL6_OOS", FINAL_TARGET_VOL
    )

    dev10 = rot.metrics(dev10_t, dev10_e)
    oos10 = rot.metrics(oos10_t, oos10_e)
    dev6 = rot.metrics(dev6_t, dev6_e)
    oos6 = rot.metrics(oos6_t, oos6_e)

    ydev = rot.yearly_stats(dev6_t)
    evaluable = ydev[ydev["trades"] >= 12].copy()

    profitable_year_ratio = (
        float((evaluable["net_pnl"] > 0).mean()) if len(evaluable) else 0.0
    )

    symbols = rot.per_symbol(dev6_t)
    nonnegative, concentration = rot.concentration_and_breadth(symbols)

    development_pass = bool(
        dev6["trades"] >= 200
        and dev6["return_pct"] > 0
        and dev6["win_rate_pct"] >= 52.0
        and dev6["profit_factor"] >= 1.20
        and dev6["expectancy"] > 0
        and dev6["max_drawdown_pct"] <= 12.0
        and profitable_year_ratio >= 0.625
        and nonnegative >= 8
        and concentration <= 0.35
    )

    oos_pass = bool(
        oos6["trades"] >= 50
        and oos6["return_pct"] > 0
        and oos6["win_rate_pct"] >= 52.0
        and oos6["profit_factor"] >= 1.20
        and oos6["expectancy"] > 0
        and oos6["max_drawdown_pct"] <= 10.0
    )

    drawdown_improved = bool(
        dev6["max_drawdown_pct"] < dev10["max_drawdown_pct"]
        and oos6["max_drawdown_pct"] < oos10["max_drawdown_pct"]
    )

    eligible = development_pass and oos_pass and drawdown_improved

    summary = {
        "test": "METALS_ROTATION_FINAL_6PCT_VOL_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_target_annual_volatility": FINAL_TARGET_VOL,
        "target_selected_before_results": True,
        "no_parameter_search": True,
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
            "vol10_reference": dev10,
            "vol6_final": dev6,
            "vol6_profitable_year_ratio": round(profitable_year_ratio, 3),
            "vol6_evaluable_years": int(len(evaluable)),
            "vol6_nonnegative_symbols": int(nonnegative),
            "vol6_positive_pnl_concentration": round(float(concentration), 4),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
            "vol10_reference": oos10,
            "vol6_final": oos6,
        },
        "development_pass": development_pass,
        "oos_diagnostic_pass": oos_pass,
        "drawdown_improved_vs_10pct": drawdown_improved,
        "eligible_for_fresh_forward_paper": eligible,
        "decision": (
            "PASS: FREEZE STRATEGY AND START FRESH FORWARD PAPER"
            if eligible
            else "FAIL: DO NOT ADVANCE TO FORWARD PAPER"
        ),
        "real_money_status": "NOT APPROVED",
        "forward_rules_if_pass": {
            "minimum_calendar_months": 6,
            "minimum_completed_position_periods": 40,
            "win_rate_min_pct": 52.0,
            "profit_factor_min": 1.25,
            "expectancy_positive": True,
            "max_drawdown_pct": 10.0,
            "parameters_must_remain_frozen": True,
            "no_real_money_before_forward_gate": True,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"target_vol": 0.10, "period": "Development", **dev10},
        {"target_vol": 0.06, "period": "Development", **dev6},
        {"target_vol": 0.10, "period": "OOS_diagnostic", **oos10},
        {"target_vol": 0.06, "period": "OOS_diagnostic", **oos6},
    ]).to_csv(OUT_COMPARISON, index=False)

    ydev.to_csv(OUT_YEARLY, index=False)

    trade_rows = []
    for label, trades in [
        ("VOL6_Development", dev6_t),
        ("VOL6_OOS", oos6_t),
    ]:
        for t in trades:
            row = asdict(t)
            row["validation_period"] = label
            trade_rows.append(row)

    pd.DataFrame(trade_rows).to_csv(OUT_TRADES, index=False)
    oos6_e.to_csv(OUT_EQUITY, index=False)

    pd.concat([dev6_x, oos6_x], ignore_index=True).to_csv(
        OUT_EXPOSURE, index=False
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
