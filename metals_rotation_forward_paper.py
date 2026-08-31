#!/usr/bin/env python3
"""
METALS ROTATION — FRESH FORWARD PAPER

FROZEN strategy. No optimization after 2026-08-31.

Fresh forward start:
    2026-08-31

This script reconstructs the paper account from the fixed forward start
on every run. It never places real orders and never needs API keys.

Frozen strategy:
- Universe/ranking from research_metals_rotation.py
- Rebalance every 21 trading sessions
- Top 4 eligible assets
- SPY > SMA200 regime filter
- Asset > SMA200, 63d momentum > 0, 126d momentum > 0
- Score = 0.5*ret63 + 0.5*ret126
- 6% annualized volatility target
- 63-session covariance/volatility lookback
- no leverage
- unused capital remains cash
- fees 0.05% per side
- slippage 0.05% per side
- starting paper capital EUR 1,000

Forward gate (pre-registered before observing forward results):
- minimum 6 calendar months
- minimum 40 completed position-periods
- win rate >= 52%
- Profit Factor >= 1.25
- expectancy > 0
- max drawdown <= 10%
- parameters remain frozen

Passing the forward gate is necessary for considering a tiny live test,
but is not a guarantee of future profit.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research_metals_rotation as rot
import research_metals_rotation_risk as risk


FORWARD_START = pd.Timestamp("2026-08-31")
STARTING_CAPITAL = 1000.0
TARGET_VOL = 0.06

MIN_MONTHS = 6
MIN_CLOSED_TRADES = 40
MIN_WIN_RATE = 52.0
MIN_PROFIT_FACTOR = 1.25
MAX_DRAWDOWN = 10.0

OUT_SUMMARY = Path("metals_rotation_forward_summary.json")
OUT_TRADES = Path("metals_rotation_forward_trades.csv")
OUT_EQUITY = Path("metals_rotation_forward_equity.csv")
OUT_POSITIONS = Path("metals_rotation_forward_positions.csv")
OUT_EXPOSURE = Path("metals_rotation_forward_exposure.csv")
OUT_REPORT = Path("metals_rotation_forward_report.md")


def metrics(trades, equity_df):
    if not trades:
        return {
            "closed_trades": 0,
            "realized_pnl": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": (
                round(float(equity_df["drawdown_pct"].max()), 3)
                if len(equity_df) else 0.0
            ),
        }

    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
    gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "closed_trades": int(len(trades)),
        "realized_pnl": round(float(pnl.sum()), 2),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(pnl.mean()), 3),
        "max_drawdown_pct": round(
            float(equity_df["drawdown_pct"].max()) if len(equity_df) else 0.0,
            3,
        ),
    }


def simulate_forward(features, start_dt, end_dt):
    dates = rot.common_dates(features, start_dt, end_dt)

    cash = STARTING_CAPITAL
    positions = {}
    entries = {}
    trades = []
    equity_rows = []
    exposure_rows = []

    peak = STARTING_CAPITAL
    bars_since_rebalance = rot.REBALANCE_EVERY
    pending = None

    if not dates:
        return {
            "dates": [],
            "cash": cash,
            "positions": positions,
            "entries": entries,
            "trades": trades,
            "equity": pd.DataFrame(),
            "exposure": pd.DataFrame(),
            "pending": pending,
            "bars_since_rebalance": bars_since_rebalance,
        }

    old_target = risk.TARGET_VOL
    risk.TARGET_VOL = TARGET_VOL

    try:
        for i, d in enumerate(dates):
            # 1) Execute the target generated at the PRIOR close.
            if pending is not None:
                target = list(pending["symbols"])
                gross = float(pending["gross"])
                est_vol = pending["estimated_vol"]
                signal_date = pending["signal_date"]

                # Close old portfolio at today's open.
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
                        period="FORWARD_PAPER",
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

                # Open new portfolio using risk-targeted gross exposure.
                if target and gross > 0:
                    equity_for_allocation = cash
                    invested_capital = equity_for_allocation * gross
                    allocation = invested_capital / len(target)

                    for sym in target:
                        df = features[sym]
                        if d not in df.index:
                            continue

                        open_px = float(df.loc[d, "open"])
                        entry_fill = open_px * (1.0 + rot.SLIPPAGE_RATE)

                        qty = allocation / (
                            entry_fill * (1.0 + rot.FEE_RATE)
                        )

                        if qty <= 0:
                            continue

                        entry_fee = qty * entry_fill * rot.FEE_RATE
                        outlay = qty * entry_fill + entry_fee

                        if outlay > cash:
                            qty = cash / (
                                entry_fill * (1.0 + rot.FEE_RATE)
                            )
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
                    "signal_date": signal_date,
                    "execution_date": d.strftime("%Y-%m-%d"),
                    "symbols": ",".join(target),
                    "estimated_annual_vol": (
                        round(float(est_vol), 6)
                        if est_vol is not None else None
                    ),
                    "gross_exposure": round(gross, 6),
                })

                pending = None
                bars_since_rebalance = 0

            # 2) Mark current account to market at close.
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

            # 3) Generate a new rebalance target at today's close.
            bars_since_rebalance += 1

            if (
                bars_since_rebalance >= rot.REBALANCE_EVERY
                and i < len(dates)
            ):
                # On the latest date this target remains pending until
                # a future run contains the next trading day's open.
                target = rot.rank_assets(features, d)

                if target:
                    gross, estimated_vol = risk.target_gross(
                        features, target, d
                    )
                else:
                    gross, estimated_vol = 0.0, None

                pending = {
                    "symbols": target,
                    "gross": gross,
                    "estimated_vol": estimated_vol,
                    "signal_date": d.strftime("%Y-%m-%d"),
                }

                # If another trading date exists inside this same run,
                # the pending target should execute on that next date.
                # The next loop iteration does exactly that.

    finally:
        risk.TARGET_VOL = old_target

    return {
        "dates": dates,
        "cash": cash,
        "positions": positions,
        "entries": entries,
        "trades": trades,
        "equity": pd.DataFrame(equity_rows),
        "exposure": pd.DataFrame(exposure_rows),
        "pending": pending,
        "bars_since_rebalance": bars_since_rebalance,
    }


def current_positions_table(state, features):
    if not state["dates"] or not state["positions"]:
        return pd.DataFrame(columns=[
            "symbol", "qty", "entry_date", "entry_price",
            "latest_close", "market_value", "unrealized_pnl_if_closed_now"
        ])

    d = state["dates"][-1]
    rows = []

    for sym, qty in state["positions"].items():
        ent = state["entries"][sym]
        df = features[sym]

        if d in df.index:
            latest = float(df.loc[d, "close"])
        else:
            avail = df.index[df.index <= d]
            latest = (
                float(df.loc[avail[-1], "close"])
                if len(avail) else ent["entry"]
            )

        hypothetical_exit = latest * (1.0 - rot.SLIPPAGE_RATE)
        exit_fee = qty * hypothetical_exit * rot.FEE_RATE
        proceeds = qty * hypothetical_exit - exit_fee
        entry_outlay = qty * ent["entry"] + ent["entry_fee"]
        unrealized = proceeds - entry_outlay

        rows.append({
            "symbol": sym,
            "qty": qty,
            "entry_date": ent["date"],
            "entry_price": round(float(ent["entry"]), 6),
            "latest_close": round(latest, 6),
            "market_value": round(qty * latest, 2),
            "unrealized_pnl_if_closed_now": round(unrealized, 2),
        })

    return pd.DataFrame(rows)


def months_elapsed(start, latest):
    if latest < start:
        return 0.0
    return round((latest - start).days / 30.4375, 2)


def main():
    data = rot.download_data()
    features = rot.build_features(data)

    latest_available = max(df.index.max() for df in features.values())

    state = simulate_forward(
        features,
        FORWARD_START,
        latest_available,
    )

    eq = state["equity"]
    trades = state["trades"]
    positions_df = current_positions_table(state, features)
    m = metrics(trades, eq)

    latest_forward_date = (
        state["dates"][-1] if state["dates"] else None
    )

    if len(eq):
        total_equity = float(eq.iloc[-1]["equity"])
    else:
        total_equity = STARTING_CAPITAL

    unrealized = (
        float(positions_df["unrealized_pnl_if_closed_now"].sum())
        if len(positions_df) else 0.0
    )

    elapsed_months = (
        months_elapsed(FORWARD_START, latest_forward_date)
        if latest_forward_date is not None else 0.0
    )

    time_ready = elapsed_months >= MIN_MONTHS
    trades_ready = m["closed_trades"] >= MIN_CLOSED_TRADES

    performance_ready = bool(
        m["win_rate_pct"] >= MIN_WIN_RATE
        and m["profit_factor"] >= MIN_PROFIT_FACTOR
        and m["expectancy"] > 0
        and m["max_drawdown_pct"] <= MAX_DRAWDOWN
    )

    forward_gate_pass = bool(
        time_ready and trades_ready and performance_ready
    )

    pending = state["pending"]

    summary = {
        "strategy": "METALS_ROTATION_FROZEN_6PCT_VOL_FORWARD_PAPER",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "real_money_status": "NOT APPROVED",
        "forward_start": str(FORWARD_START.date()),
        "latest_market_date": (
            str(latest_forward_date.date())
            if latest_forward_date is not None else None
        ),
        "starting_capital_eur": STARTING_CAPITAL,
        "current_equity_eur": round(total_equity, 2),
        "total_return_pct": round(
            (total_equity / STARTING_CAPITAL - 1.0) * 100.0, 3
        ),
        "cash_eur": round(float(state["cash"]), 2),
        "open_positions": int(len(state["positions"])),
        "unrealized_pnl_if_closed_now_eur": round(unrealized, 2),
        **m,
        "months_elapsed": elapsed_months,
        "pending_rebalance": (
            {
                "signal_date": pending["signal_date"],
                "symbols": pending["symbols"],
                "gross_exposure": round(float(pending["gross"]), 6),
                "estimated_annual_vol": (
                    round(float(pending["estimated_vol"]), 6)
                    if pending["estimated_vol"] is not None else None
                ),
            }
            if pending is not None else None
        ),
        "forward_gate": {
            "minimum_calendar_months": MIN_MONTHS,
            "minimum_closed_position_periods": MIN_CLOSED_TRADES,
            "minimum_win_rate_pct": MIN_WIN_RATE,
            "minimum_profit_factor": MIN_PROFIT_FACTOR,
            "expectancy_must_be_positive": True,
            "maximum_drawdown_pct": MAX_DRAWDOWN,
            "time_ready": time_ready,
            "trades_ready": trades_ready,
            "performance_ready": performance_ready,
            "pass": forward_gate_pass,
        },
        "parameters_frozen": {
            "target_annual_volatility": TARGET_VOL,
            "volatility_lookback_sessions": risk.VOL_LOOKBACK,
            "rebalance_every_sessions": rot.REBALANCE_EVERY,
            "top_n": rot.TOP_N,
            "momentum_fast_sessions": rot.LOOKBACK_FAST,
            "momentum_slow_sessions": rot.LOOKBACK_SLOW,
            "trend_sma_sessions": rot.SMA_TREND,
            "fee_rate_per_side": rot.FEE_RATE,
            "slippage_rate_per_side": rot.SLIPPAGE_RATE,
            "leverage": False,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([asdict(t) for t in trades]).to_csv(
        OUT_TRADES, index=False
    )

    eq.to_csv(OUT_EQUITY, index=False)
    positions_df.to_csv(OUT_POSITIONS, index=False)
    state["exposure"].to_csv(OUT_EXPOSURE, index=False)

    pending_text = "No"
    if pending is not None:
        pending_text = (
            f"Sí — {', '.join(pending['symbols']) or 'cash'} "
            f"(gross {pending['gross']:.1%})"
        )

    report = f"""# Metals Rotation — Forward Paper

**PAPER ONLY — sin órdenes reales**

- Inicio forward: **{FORWARD_START.date()}**
- Último dato: **{summary['latest_market_date']}**
- Capital inicial: **€{STARTING_CAPITAL:.2f}**
- Equity actual: **€{total_equity:.2f}**
- Return total: **{summary['total_return_pct']:.3f}%**
- Operaciones/periodos cerrados: **{m['closed_trades']}**
- PnL realizado: **€{m['realized_pnl']:.2f}**
- Win rate: **{m['win_rate_pct']:.2f}%**
- Profit Factor: **{m['profit_factor']:.3f}**
- Expectancy: **€{m['expectancy']:.3f}/periodo**
- Max DD: **{m['max_drawdown_pct']:.3f}%**
- Posiciones abiertas: **{len(state['positions'])}**
- Rebalance pendiente: **{pending_text}**

## Gate de forward

- Meses transcurridos: **{elapsed_months:.2f} / {MIN_MONTHS}**
- Periodos cerrados: **{m['closed_trades']} / {MIN_CLOSED_TRADES}**
- Gate completo: **{'PASS' if forward_gate_pass else 'AÚN NO'}**

Los parámetros están congelados desde el inicio del forward.
No se autoriza dinero real desde este informe.
"""

    OUT_REPORT.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
