#!/usr/bin/env python3
"""
METALS ROTATION — VOLATILITY RISK CONTROL

The base rotation strategy showed a real positive edge:
- >50% winning position-periods
- positive Development and OOS PnL
- broad yearly profitability

Its weakness was drawdown, caused by putting 100% of capital into a
small set of volatile/correlated assets.

This test does NOT change:
- universe
- ranking
- momentum lookbacks
- SPY regime rule
- top-4 selection
- 21-trading-day rebalance cadence
- fees/slippage

It adds ONE standard risk overlay:
- estimate the selected equal-weight portfolio's trailing 63-day
  annualized volatility at each rebalance close
- target 10% annualized portfolio volatility
- gross exposure = min(100%, 10% / estimated portfolio volatility)
- unallocated capital remains in cash

No leverage. No parameter search. OOS is diagnostic only.
Passing can authorize only fresh forward PAPER trading.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research_metals_rotation as rot


VOL_LOOKBACK = 63
TARGET_VOL = 0.10

OUT_SUMMARY = Path("metals_rotation_risk_summary.json")
OUT_COMPARISON = Path("metals_rotation_risk_comparison.csv")
OUT_YEARLY = Path("metals_rotation_risk_yearly.csv")
OUT_TRADES = Path("metals_rotation_risk_trades.csv")
OUT_EQUITY = Path("metals_rotation_risk_oos_equity.csv")
OUT_EXPOSURE = Path("metals_rotation_risk_exposure.csv")


def estimate_equal_weight_vol(features, symbols, d):
    if not symbols:
        return None

    series = []

    for sym in symbols:
        df = features[sym]
        hist = df.loc[df.index <= d, "close"].pct_change().rename(sym)
        series.append(hist)

    rets = pd.concat(series, axis=1).dropna().tail(VOL_LOOKBACK)

    if len(rets) < 40:
        return None

    n = len(symbols)
    w = np.repeat(1.0 / n, n)
    cov = rets.cov().values * 252.0

    var = float(w @ cov @ w)
    if not np.isfinite(var) or var <= 0:
        return None

    return float(np.sqrt(var))


def target_gross(features, symbols, d):
    vol = estimate_equal_weight_vol(features, symbols, d)

    if vol is None or vol <= 0:
        return 0.0, None

    gross = min(1.0, TARGET_VOL / vol)
    gross = max(0.0, gross)
    return gross, vol


def simulate_risk_control(period, features, start_dt, end_dt):
    dates = rot.common_dates(features, start_dt, end_dt)

    if len(dates) < 3:
        return [], pd.DataFrame(), pd.DataFrame()

    cash = rot.STARTING_CAPITAL
    positions = {}
    entries = {}
    trades = []
    equity_rows = []
    exposure_rows = []

    peak = rot.STARTING_CAPITAL
    bars_since_rebalance = rot.REBALANCE_EVERY
    pending = None

    for i, d in enumerate(dates):
        # 1) Execute target generated at prior close.
        if pending is not None:
            target = pending["symbols"]
            gross = float(pending["gross"])
            estimated_vol = pending["estimated_vol"]
            signal_date = pending["signal_date"]

            # Liquidate old portfolio at today's open.
            for sym in list(positions):
                df = features[sym]
                if d not in df.index:
                    continue

                qty = positions.pop(sym)
                ent = entries.pop(sym)

                open_px = float(df.loc[d, "open"])
                exit_fill = open_px * (1.0 - rot.SLIPPAGE_RATE)
                exit_fee = qty * exit_fill * rot.FEE_RATE
                proceeds = qty * exit_fill - exit_fee
                cash += proceeds

                entry_outlay = qty * ent["entry"] + ent["entry_fee"]
                net = proceeds - entry_outlay

                trades.append(rot.Trade(
                    period=period,
                    symbol=sym,
                    entry_date=ent["date"],
                    exit_date=d.strftime("%Y-%m-%d"),
                    entry=ent["entry"],
                    exit=exit_fill,
                    qty=qty,
                    net_pnl=net,
                    return_pct=(exit_fill / ent["entry"] - 1.0) * 100.0,
                    hold_days=int((d - pd.Timestamp(ent["date"])).days),
                    exit_reason="REBALANCE",
                ))

            # Allocate only the risk-targeted fraction; rest stays in cash.
            if target and gross > 0:
                total_equity = cash
                invested_capital = total_equity * gross
                allocation = invested_capital / len(target)

                for sym in target:
                    df = features[sym]
                    if d not in df.index:
                        continue

                    open_px = float(df.loc[d, "open"])
                    entry_fill = open_px * (1.0 + rot.SLIPPAGE_RATE)
                    qty = allocation / (entry_fill * (1.0 + rot.FEE_RATE))

                    if qty <= 0:
                        continue

                    entry_fee = qty * entry_fill * rot.FEE_RATE
                    outlay = qty * entry_fill + entry_fee

                    if outlay > cash:
                        qty = cash / (entry_fill * (1.0 + rot.FEE_RATE))
                        entry_fee = qty * entry_fill * rot.FEE_RATE
                        outlay = qty * entry_fill + entry_fee

                    if qty <= 0 or outlay <= 0:
                        continue

                    cash -= outlay
                    positions[sym] = qty
                    entries[sym] = {
                        "entry": entry_fill,
                        "entry_fee": entry_fee,
                        "date": d.strftime("%Y-%m-%d"),
                    }

            exposure_rows.append({
                "period": period,
                "signal_date": signal_date,
                "execution_date": d.strftime("%Y-%m-%d"),
                "symbols": ",".join(target),
                "estimated_annual_vol": (
                    round(float(estimated_vol), 6)
                    if estimated_vol is not None else None
                ),
                "gross_exposure": round(gross, 6),
            })

            pending = None
            bars_since_rebalance = 0

        # 2) Mark to market.
        equity = cash

        for sym, qty in positions.items():
            df = features[sym]
            mark = (
                float(df.loc[d, "close"])
                if d in df.index
                else entries[sym]["entry"]
            )
            equity += qty * mark

        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0

        equity_rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "drawdown_pct": dd,
        })

        # 3) Generate same rotation target, plus risk scaling.
        bars_since_rebalance += 1

        if bars_since_rebalance >= rot.REBALANCE_EVERY and i < len(dates) - 1:
            target = rot.rank_assets(features, d)

            if target:
                gross, estimated_vol = target_gross(features, target, d)
            else:
                gross, estimated_vol = 0.0, None

            pending = {
                "symbols": target,
                "gross": gross,
                "estimated_vol": estimated_vol,
                "signal_date": d.strftime("%Y-%m-%d"),
            }

    # Force close final positions for reporting.
    last = dates[-1]

    for sym in list(positions):
        qty = positions.pop(sym)
        ent = entries.pop(sym)

        df = features[sym]
        avail = df.index[df.index <= last]
        if not len(avail):
            continue

        d = avail[-1]
        exit_fill = float(df.loc[d, "close"]) * (1.0 - rot.SLIPPAGE_RATE)
        exit_fee = qty * exit_fill * rot.FEE_RATE
        proceeds = qty * exit_fill - exit_fee
        cash += proceeds

        entry_outlay = qty * ent["entry"] + ent["entry_fee"]
        net = proceeds - entry_outlay

        trades.append(rot.Trade(
            period=period,
            symbol=sym,
            entry_date=ent["date"],
            exit_date=d.strftime("%Y-%m-%d"),
            entry=ent["entry"],
            exit=exit_fill,
            qty=qty,
            net_pnl=net,
            return_pct=(exit_fill / ent["entry"] - 1.0) * 100.0,
            hold_days=int((d - pd.Timestamp(ent["date"])).days),
            exit_reason="END_OF_TEST",
        ))

    return trades, pd.DataFrame(equity_rows), pd.DataFrame(exposure_rows)


def yearly_with_label(label, trades):
    df = rot.yearly_stats(trades)
    if df.empty:
        return df
    df.insert(0, "variant", label)
    return df


def profitable_year_ratio(year_df):
    if year_df.empty:
        return 0.0, 0, 0

    evaluable = year_df[year_df["trades"] >= 12]

    if evaluable.empty:
        return 0.0, 0, 0

    positive = int((evaluable["net_pnl"] > 0).sum())
    total = int(len(evaluable))

    return positive / total, positive, total


def main():
    data = rot.download_data()
    features = rot.build_features(data)

    end = max(df.index.max() for df in features.values())
    oos_start = (end - pd.DateOffset(months=rot.OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(rot.START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    # BASE for exact comparison.
    base_dev_trades, base_dev_eq = rot.simulate(
        "BASE_Development", features, dev_start, dev_end
    )
    base_oos_trades, base_oos_eq = rot.simulate(
        "BASE_OOS", features, oos_start, end
    )

    # Fixed volatility overlay.
    risk_dev_trades, risk_dev_eq, risk_dev_exposure = simulate_risk_control(
        "RISK_Development", features, dev_start, dev_end
    )
    risk_oos_trades, risk_oos_eq, risk_oos_exposure = simulate_risk_control(
        "RISK_OOS", features, oos_start, end
    )

    base_dev = rot.metrics(base_dev_trades, base_dev_eq)
    base_oos = rot.metrics(base_oos_trades, base_oos_eq)
    risk_dev = rot.metrics(risk_dev_trades, risk_dev_eq)
    risk_oos = rot.metrics(risk_oos_trades, risk_oos_eq)

    y_base_dev = yearly_with_label("BASE_Development", base_dev_trades)
    y_risk_dev = yearly_with_label("RISK_Development", risk_dev_trades)
    y_base_oos = yearly_with_label("BASE_OOS", base_oos_trades)
    y_risk_oos = yearly_with_label("RISK_OOS", risk_oos_trades)

    ratio, positive_years, eval_years = profitable_year_ratio(
        y_risk_dev.drop(columns=["variant"]) if not y_risk_dev.empty else y_risk_dev
    )

    risk_symbols = rot.per_symbol(risk_dev_trades)
    nonnegative, concentration = rot.concentration_and_breadth(risk_symbols)

    development_pass = bool(
        risk_dev["trades"] >= 200
        and risk_dev["return_pct"] > 0
        and risk_dev["win_rate_pct"] >= 52.0
        and risk_dev["profit_factor"] >= 1.20
        and risk_dev["expectancy"] > 0
        and risk_dev["max_drawdown_pct"] <= 12.0
        and ratio >= 0.625
        and nonnegative >= 8
        and concentration <= 0.35
    )

    oos_diagnostic_pass = bool(
        risk_oos["trades"] >= 50
        and risk_oos["return_pct"] > 0
        and risk_oos["win_rate_pct"] >= 52.0
        and risk_oos["profit_factor"] >= 1.20
        and risk_oos["expectancy"] > 0
        and risk_oos["max_drawdown_pct"] <= 10.0
    )

    # Risk overlay should materially improve drawdown vs base.
    risk_improvement = bool(
        risk_dev["max_drawdown_pct"] < base_dev["max_drawdown_pct"]
        and risk_oos["max_drawdown_pct"] < base_oos["max_drawdown_pct"]
    )

    eligible = development_pass and oos_diagnostic_pass and risk_improvement

    summary = {
        "test": "METALS_ROTATION_10PCT_VOLATILITY_RISK_CONTROL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "risk_rule": {
            "lookback_trading_days": VOL_LOOKBACK,
            "target_annual_volatility": TARGET_VOL,
            "gross_exposure_rule": "min(1.0, target_vol / estimated_equal_weight_portfolio_vol)",
            "leverage": False,
            "cash_for_unallocated_capital": True,
        },
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
            "base": base_dev,
            "risk_control": risk_dev,
            "risk_control_profitable_year_ratio": round(ratio, 3),
            "risk_control_positive_years": positive_years,
            "risk_control_evaluable_years": eval_years,
            "risk_control_nonnegative_symbols": nonnegative,
            "risk_control_positive_pnl_concentration": round(concentration, 4),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
            "base": base_oos,
            "risk_control": risk_oos,
        },
        "development_pass": development_pass,
        "oos_diagnostic_pass": oos_diagnostic_pass,
        "drawdown_improved_vs_base": risk_improvement,
        "eligible_for_fresh_forward_paper": eligible,
        "decision": (
            "ELIGIBLE: freeze rotation + risk overlay and start fresh forward paper"
            if eligible
            else "FAIL: do not advance yet"
        ),
        "real_money_status": "NOT APPROVED",
        "future_forward_gate_if_eligible": {
            "minimum_calendar_months": 6,
            "minimum_completed_position_periods": 40,
            "win_rate_min_pct": 52.0,
            "profit_factor_min": 1.25,
            "expectancy_positive": True,
            "max_drawdown_pct": 10.0,
            "parameters_frozen": True,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"variant": "BASE", "period": "Development", **base_dev},
        {"variant": "RISK_CONTROL", "period": "Development", **risk_dev},
        {"variant": "BASE", "period": "OOS_diagnostic", **base_oos},
        {"variant": "RISK_CONTROL", "period": "OOS_diagnostic", **risk_oos},
    ]).to_csv(OUT_COMPARISON, index=False)

    pd.concat(
        [x for x in [y_base_dev, y_risk_dev, y_base_oos, y_risk_oos] if not x.empty],
        ignore_index=True,
    ).to_csv(OUT_YEARLY, index=False)

    all_trades = []
    for label, ts in [
        ("BASE_Development", base_dev_trades),
        ("RISK_Development", risk_dev_trades),
        ("BASE_OOS", base_oos_trades),
        ("RISK_OOS", risk_oos_trades),
    ]:
        for t in ts:
            row = asdict(t)
            row["variant_period"] = label
            all_trades.append(row)

    pd.DataFrame(all_trades).to_csv(OUT_TRADES, index=False)

    risk_oos_eq.to_csv(OUT_EQUITY, index=False)

    pd.concat(
        [risk_dev_exposure, risk_oos_exposure],
        ignore_index=True,
    ).to_csv(OUT_EXPOSURE, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
