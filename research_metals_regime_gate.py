#!/usr/bin/env python3
"""
ROBUSTNESS GATE — METALS & STOCKS

This is NOT a new optimized strategy family.

We compare exactly two pre-declared versions:
1) BASE: the already-tested fixed strategy.
2) SECTOR_GATE: same strategy, same entries/exits/risk, but mining stocks
   are allowed to open a new position only while the broad metals/mining
   ETF XME is in an established uptrend:
       XME close > EMA200
       XME EMA50 > EMA200

Non-mining stocks keep the original rules.

Why:
The first research showed that the long-term weakness was concentrated
mainly in mining names during hostile sector regimes. This gate is
economically interpretable and does not change stops, breakout length,
risk, holding period, fees, or symbol selection.

Important:
- Candidate selection is judged primarily on Development.
- The already-seen OOS period is only a diagnostic confirmation.
- Real-money approval is NOT possible from this script.
- If eligible, the next stage is fresh forward paper trading from a
  fixed future start date with parameters frozen.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import research_metals_stocks as base


MINERS = {"FCX", "SCCO", "NEM", "AEM", "WPM", "CCJ", "BHP", "RIO"}
SECTOR_BENCHMARK = "XME"

OUT_SUMMARY = Path("metals_regime_gate_summary.json")
OUT_YEARLY = Path("metals_regime_gate_yearly.csv")
OUT_OOS_SYMBOLS = Path("metals_regime_gate_oos_symbols.csv")
OUT_COMPARISON = Path("metals_regime_gate_comparison.csv")
OUT_TRADES = Path("metals_regime_gate_trades.csv")


def normalize_yf(raw, symbol):
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can use (Price, Ticker) or (Ticker, Price)
        levels0 = set(map(str, df.columns.get_level_values(0)))
        levels1 = set(map(str, df.columns.get_level_values(1)))
        if symbol in levels1:
            df = df.xs(symbol, axis=1, level=1)
        elif symbol in levels0:
            df = df.xs(symbol, axis=1, level=0)

    df.columns = [str(c).lower() for c in df.columns]

    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        raise RuntimeError(f"Bad columns for {symbol}: {list(df.columns)}")

    df = df[needed].dropna(subset=["open", "high", "low", "close"]).copy()

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx

    return df


def load_xme():
    raw = yf.download(
        SECTOR_BENCHMARK,
        start=base.START_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    xme = normalize_yf(raw, SECTOR_BENCHMARK)
    xme["ema50"] = xme["close"].ewm(span=base.EMA_FAST, adjust=False).mean()
    xme["ema200"] = xme["close"].ewm(span=base.EMA_SLOW, adjust=False).mean()
    xme["sector_on"] = (
        (xme["close"] > xme["ema200"])
        & (xme["ema50"] > xme["ema200"])
    )
    return xme


def build_prepared(data, xme):
    prepared = {}

    for sym in base.TRADE_SYMBOLS:
        if sym not in data:
            continue

        df = base.prepare(data[sym], data[base.BENCHMARK]).copy()

        xme_on = xme["sector_on"].reindex(df.index).ffill().fillna(False)
        df["_xme_sector_on"] = xme_on.astype(bool)
        df["_is_miner"] = sym in MINERS

        prepared[sym] = df

    return prepared


def run_variant(name, prepared, dev_start, dev_end, oos_start, oos_end, signal_function):
    original = base.signal_ok
    base.signal_ok = signal_function
    try:
        dev_trades, dev_eq = base.simulate(
            f"{name}_Development", prepared, dev_start, dev_end
        )
        oos_trades, oos_eq = base.simulate(
            f"{name}_OOS", prepared, oos_start, oos_end
        )
    finally:
        base.signal_ok = original

    return {
        "dev_trades": dev_trades,
        "dev_eq": dev_eq,
        "oos_trades": oos_trades,
        "oos_eq": oos_eq,
        "dev_metrics": base.metrics(dev_trades, dev_eq),
        "oos_metrics": base.metrics(oos_trades, oos_eq),
    }


def yearly_stats(label, trades):
    if not trades:
        return pd.DataFrame(columns=[
            "variant", "year", "trades", "net_pnl",
            "win_rate_pct", "profit_factor", "expectancy"
        ])

    df = pd.DataFrame([vars(t) for t in trades])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    rows = []
    for year, g in df.groupby("year"):
        pnl = g["net_pnl"].astype(float)
        gp = float(pnl[pnl > 0].sum())
        gl = abs(float(pnl[pnl < 0].sum()))
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "variant": label,
            "year": int(year),
            "trades": int(len(g)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def profitable_year_ratio(year_df):
    # Ignore years with < 5 closed trades; too little information.
    x = year_df[year_df["trades"] >= 5].copy()
    if x.empty:
        return 0.0, 0, 0
    positive = int((x["net_pnl"] > 0).sum())
    total = int(len(x))
    return positive / total, positive, total


def candidate_signal(original_signal):
    def _signal(row):
        if not original_signal(row):
            return False

        if bool(row["_is_miner"]):
            return bool(row["_xme_sector_on"])

        return True

    return _signal


def miner_metrics(trades):
    ts = [t for t in trades if t.symbol in MINERS]
    if not ts:
        return {"trades": 0, "net_pnl": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0}

    pnl = np.array([t.net_pnl for t in ts], dtype=float)
    gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
    gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "trades": int(len(ts)),
        "net_pnl": round(float(pnl.sum()), 2),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(pnl.mean()), 3),
    }


def main():
    data = base.download_data()
    xme = load_xme()
    prepared = build_prepared(data, xme)

    end = max(df.index.max() for df in prepared.values())
    oos_start = (end - pd.DateOffset(months=base.OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(base.START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    original_signal = base.signal_ok

    base_result = run_variant(
        "BASE",
        prepared,
        dev_start,
        dev_end,
        oos_start,
        end,
        original_signal,
    )

    gated_result = run_variant(
        "SECTOR_GATE",
        prepared,
        dev_start,
        dev_end,
        oos_start,
        end,
        candidate_signal(original_signal),
    )

    y_base_dev = yearly_stats("BASE_Development", base_result["dev_trades"])
    y_gate_dev = yearly_stats("SECTOR_GATE_Development", gated_result["dev_trades"])
    y_base_oos = yearly_stats("BASE_OOS", base_result["oos_trades"])
    y_gate_oos = yearly_stats("SECTOR_GATE_OOS", gated_result["oos_trades"])
    yearly = pd.concat([y_base_dev, y_gate_dev, y_base_oos, y_gate_oos], ignore_index=True)

    gate_year_ratio, gate_positive_years, gate_years = profitable_year_ratio(y_gate_dev)

    bdev = base_result["dev_metrics"]
    gdev = gated_result["dev_metrics"]
    boos = base_result["oos_metrics"]
    goos = gated_result["oos_metrics"]

    # Development is the real robustness test for the new gate.
    development_robust = bool(
        gdev["trades"] >= 70
        and gdev["net_pnl"] > 0
        and gdev["profit_factor"] >= 1.15
        and gdev["expectancy"] > 0
        and gdev["max_drawdown_pct"] <= 10.0
        and gate_year_ratio >= 0.60
        and gdev["profit_factor"] > bdev["profit_factor"]
        and gdev["net_pnl"] > bdev["net_pnl"]
    )

    # OOS is diagnostic because it has already been viewed once.
    oos_diagnostic_ok = bool(
        goos["trades"] >= 20
        and goos["net_pnl"] > 0
        and goos["profit_factor"] >= 1.25
        and goos["expectancy"] > 0
        and goos["win_rate_pct"] >= 50.0
        and goos["max_drawdown_pct"] <= 10.0
    )

    eligible_forward_paper = development_robust and oos_diagnostic_ok

    summary = {
        "test": "METALS_STOCKS_XME_SECTOR_REGIME_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "Same base strategy. Mining symbols may enter only when "
            "XME close > EMA200 and XME EMA50 > EMA200."
        ),
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
            "base": bdev,
            "sector_gate": gdev,
            "sector_gate_profitable_year_ratio": round(gate_year_ratio, 3),
            "sector_gate_positive_years": gate_positive_years,
            "sector_gate_evaluable_years": gate_years,
            "base_miners": miner_metrics(base_result["dev_trades"]),
            "sector_gate_miners": miner_metrics(gated_result["dev_trades"]),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
            "base": boos,
            "sector_gate": goos,
            "base_miners": miner_metrics(base_result["oos_trades"]),
            "sector_gate_miners": miner_metrics(gated_result["oos_trades"]),
        },
        "development_robust": development_robust,
        "oos_diagnostic_ok": oos_diagnostic_ok,
        "eligible_for_fresh_forward_paper": eligible_forward_paper,
        "decision": (
            "ELIGIBLE: freeze rules and start fresh forward paper"
            if eligible_forward_paper
            else "FAIL: do not advance this gate to forward paper"
        ),
        "future_real_money_gate_pre_registered": {
            "minimum_calendar_months": 6,
            "minimum_closed_forward_trades": 25,
            "profit_factor_min": 1.30,
            "win_rate_min_pct": 50.0,
            "expectancy_must_be_positive": True,
            "max_drawdown_max_pct": 10.0,
            "single_symbol_positive_pnl_concentration_max_pct": 40.0,
            "parameters_must_remain_frozen": True,
            "note": "Passing these thresholds is necessary, not a guarantee of future profit."
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    yearly.to_csv(OUT_YEARLY, index=False)

    sym = base.per_symbol(gated_result["oos_trades"])
    sym.to_csv(OUT_OOS_SYMBOLS, index=False)

    comparison = pd.DataFrame([
        {"variant": "BASE", "period": "Development", **bdev},
        {"variant": "SECTOR_GATE", "period": "Development", **gdev},
        {"variant": "BASE", "period": "OOS_diagnostic", **boos},
        {"variant": "SECTOR_GATE", "period": "OOS_diagnostic", **goos},
    ])
    comparison.to_csv(OUT_COMPARISON, index=False)

    all_trades = []
    for label, result in [("BASE", base_result), ("SECTOR_GATE", gated_result)]:
        for t in result["dev_trades"]:
            row = vars(t).copy()
            row["variant"] = label
            row["validation_period"] = "Development"
            all_trades.append(row)
        for t in result["oos_trades"]:
            row = vars(t).copy()
            row["variant"] = label
            row["validation_period"] = "OOS_diagnostic"
            all_trades.append(row)

    pd.DataFrame(all_trades).to_csv(OUT_TRADES, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
