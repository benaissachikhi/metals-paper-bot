#!/usr/bin/env python3
"""
METALS + STOCKS — MONTHLY RELATIVE-STRENGTH ROTATION

This is a NEW strategy family. We stop tuning the failed pullback family.

Goal:
Seek a positive edge with a majority of profitable holding periods while
remaining diversified and avoiding weak market regimes.

Universe:
Metals/mining ETFs:
XME, GDX, GDXJ, SIL, COPX, URA

Mining / metal stocks:
FCX, SCCO, NEM, AEM, WPM, CCJ, BHP, RIO

Large liquid stocks:
MSFT, GOOGL, AMZN, JPM

Benchmark/regime:
SPY

Rules, fixed before results:
- Long only, no leverage.
- Every 21 trading days, after the close, rank all eligible assets.
- Execute the new portfolio at the NEXT trading day's open.
- Market must be SPY > SMA200; otherwise go to cash.
- Asset must be > SMA200.
- 63-day momentum > 0.
- 126-day momentum > 0.
- Score = 50% 63-day return + 50% 126-day return.
- Hold top 4, equal weight.
- Fractional shares allowed in research.
- Fee 0.05% + slippage 0.05% per side.
- Each rebalance closes the old position-period and opens a new one.
- Starting capital EUR 1,000.

Validation:
- Data from 2018 onward.
- Last 18 months are OOS diagnostic.
- Candidate selection is NOT based on OOS.
- If it passes, next stage is FRESH forward paper only.

Development PASS:
- >= 250 completed position-period trades
- total net return > 0
- win rate >= 52%
- Profit Factor >= 1.20
- expectancy > 0
- max DD <= 15%
- profitable-year ratio >= 62.5%
- at least 8 symbols with non-negative PnL
- positive PnL concentration in one symbol <= 35%

OOS diagnostic PASS:
- >= 50 completed position-period trades
- net return > 0
- win rate >= 52%
- Profit Factor >= 1.25
- expectancy > 0
- max DD <= 12%

Passing cannot authorize real money.
"""

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


STARTING_CAPITAL = 1000.0

ASSETS = [
    "XME", "GDX", "GDXJ", "SIL", "COPX", "URA",
    "FCX", "SCCO", "NEM", "AEM", "WPM", "CCJ", "BHP", "RIO",
    "MSFT", "GOOGL", "AMZN", "JPM",
]
BENCHMARK = "SPY"

START_DATE = "2018-01-01"
OOS_MONTHS = 18

LOOKBACK_FAST = 63
LOOKBACK_SLOW = 126
SMA_TREND = 200
REBALANCE_EVERY = 21
TOP_N = 4

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

OUT_SUMMARY = Path("metals_rotation_summary.json")
OUT_COMPARISON = Path("metals_rotation_comparison.csv")
OUT_SYMBOLS = Path("metals_rotation_oos_symbols.csv")
OUT_YEARLY = Path("metals_rotation_yearly.csv")
OUT_TRADES = Path("metals_rotation_trades.csv")
OUT_EQUITY = Path("metals_rotation_oos_equity.csv")


@dataclass
class Trade:
    period: str
    symbol: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    qty: float
    net_pnl: float
    return_pct: float
    hold_days: int
    exit_reason: str


def normalize_download(raw, symbol):
    if raw.empty:
        return None

    try:
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(map(str, raw.columns.get_level_values(0)))
            level1 = set(map(str, raw.columns.get_level_values(1)))

            if symbol in level0:
                df = raw[symbol].copy()
            elif symbol in level1:
                df = raw.xs(symbol, axis=1, level=1).copy()
            else:
                return None
        else:
            df = raw.copy()
    except Exception:
        return None

    df.columns = [str(c).lower() for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]

    if not all(c in df.columns for c in needed):
        return None

    df = df[needed].dropna(subset=["open", "close"]).copy()

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx

    if len(df) < 300:
        return None

    return df


def download_data():
    symbols = ASSETS + [BENCHMARK]

    raw = yf.download(
        symbols,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No data downloaded.")

    data = {}
    for sym in symbols:
        df = normalize_download(raw, sym)
        if df is not None:
            data[sym] = df

    missing = [s for s in symbols if s not in data]
    if missing:
        print("WARNING missing:", missing)

    if BENCHMARK not in data:
        raise RuntimeError("SPY benchmark missing.")

    available_assets = [s for s in ASSETS if s in data]
    if len(available_assets) < 14:
        raise RuntimeError(f"Too few assets available: {available_assets}")

    print("Available:", available_assets)
    return data


def build_features(data):
    features = {}

    spy = data[BENCHMARK].copy()
    spy["sma200"] = spy["close"].rolling(SMA_TREND).mean()
    spy["regime"] = spy["close"] > spy["sma200"]

    for sym in ASSETS:
        if sym not in data:
            continue

        df = data[sym].copy()
        df["sma200"] = df["close"].rolling(SMA_TREND).mean()
        df["ret63"] = df["close"].pct_change(LOOKBACK_FAST)
        df["ret126"] = df["close"].pct_change(LOOKBACK_SLOW)

        spy_regime = spy["regime"].reindex(df.index).ffill()
        df["spy_regime"] = spy_regime

        df["eligible"] = (
            (df["close"] > df["sma200"])
            & (df["ret63"] > 0)
            & (df["ret126"] > 0)
            & df["spy_regime"].fillna(False)
        )

        df["score"] = 0.5 * df["ret63"] + 0.5 * df["ret126"]
        features[sym] = df.dropna().copy()

    return features


def rank_assets(features, d):
    candidates = []

    for sym, df in features.items():
        if d not in df.index:
            continue

        row = df.loc[d]
        if bool(row["eligible"]):
            candidates.append((sym, float(row["score"])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in candidates[:TOP_N]]


def common_dates(features, start_dt, end_dt):
    dates = sorted(set().union(*[
        set(df.index[(df.index >= start_dt) & (df.index <= end_dt)])
        for df in features.values()
    ]))
    return dates


def simulate(period, features, start_dt, end_dt):
    dates = common_dates(features, start_dt, end_dt)

    if len(dates) < 3:
        return [], pd.DataFrame()

    cash = STARTING_CAPITAL
    positions = {}
    entries = {}
    trades = []
    equity_rows = []

    peak = STARTING_CAPITAL
    bars_since_rebalance = REBALANCE_EVERY
    pending_target = None

    for i, d in enumerate(dates):
        # 1) Apply portfolio target generated at previous close.
        if pending_target is not None:
            target = list(pending_target)

            # Sell every existing position at today's open.
            for sym in list(positions):
                df = features[sym]
                if d not in df.index:
                    continue

                qty = positions.pop(sym)
                ent = entries.pop(sym)

                open_px = float(df.loc[d, "open"])
                exit_fill = open_px * (1.0 - SLIPPAGE_RATE)
                exit_fee = qty * exit_fill * FEE_RATE
                proceeds = qty * exit_fill - exit_fee
                cash += proceeds

                entry_outlay = qty * ent["entry"] + ent["entry_fee"]
                net = proceeds - entry_outlay

                trades.append(Trade(
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

            # Equal weight using capital after liquidation.
            if target:
                equity_for_allocation = cash
                allocation = equity_for_allocation / len(target)

                for sym in target:
                    df = features[sym]
                    if d not in df.index:
                        continue

                    open_px = float(df.loc[d, "open"])
                    entry_fill = open_px * (1.0 + SLIPPAGE_RATE)

                    qty = allocation / (entry_fill * (1.0 + FEE_RATE))

                    if qty <= 0:
                        continue

                    entry_fee = qty * entry_fill * FEE_RATE
                    outlay = qty * entry_fill + entry_fee

                    # Numerical safety: last asset can be a few cents over.
                    if outlay > cash:
                        qty = cash / (entry_fill * (1.0 + FEE_RATE))
                        entry_fee = qty * entry_fill * FEE_RATE
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

            pending_target = None
            bars_since_rebalance = 0

        # 2) Mark to market at close.
        equity = cash
        for sym, qty in positions.items():
            df = features[sym]
            if d in df.index:
                equity += qty * float(df.loc[d, "close"])
            else:
                equity += qty * entries[sym]["entry"]

        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0

        equity_rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "drawdown_pct": dd,
        })

        # 3) Generate next target every fixed 21 trading dates.
        bars_since_rebalance += 1

        if bars_since_rebalance >= REBALANCE_EVERY and i < len(dates) - 1:
            pending_target = rank_assets(features, d)

    # Force close at last close for reporting.
    last = dates[-1]

    for sym in list(positions):
        qty = positions.pop(sym)
        ent = entries.pop(sym)

        df = features[sym]
        avail = df.index[df.index <= last]
        if not len(avail):
            continue

        d = avail[-1]
        raw = float(df.loc[d, "close"])
        exit_fill = raw * (1.0 - SLIPPAGE_RATE)
        exit_fee = qty * exit_fill * FEE_RATE
        proceeds = qty * exit_fill - exit_fee
        cash += proceeds

        entry_outlay = qty * ent["entry"] + ent["entry_fee"]
        net = proceeds - entry_outlay

        trades.append(Trade(
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

    return trades, pd.DataFrame(equity_rows)


def metrics(trades, equity):
    if not trades:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl = np.array([t.net_pnl for t in trades], dtype=float)

    gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
    gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "trades": int(len(trades)),
        "net_pnl": round(float(pnl.sum()), 2),
        "return_pct": round(float(pnl.sum() / STARTING_CAPITAL * 100.0), 3),
        "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(pnl.mean()), 3),
        "max_drawdown_pct": round(
            float(equity["drawdown_pct"].max()) if len(equity) else 0.0,
            3,
        ),
    }


def yearly_stats(trades):
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(t) for t in trades])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    rows = []

    for year, g in df.groupby("year"):
        pnl = g["net_pnl"].astype(float)
        gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
        gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "year": int(year),
            "trades": int(len(g)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def per_symbol(trades):
    rows = []

    for sym in ASSETS:
        ts = [t for t in trades if t.symbol == sym]

        if not ts:
            rows.append({
                "symbol": sym,
                "trades": 0,
                "net_pnl": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
            })
            continue

        pnl = np.array([t.net_pnl for t in ts], dtype=float)
        gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
        gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "symbol": sym,
            "trades": int(len(ts)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def concentration_and_breadth(symbol_df):
    pnl = symbol_df["net_pnl"].astype(float).values
    nonnegative = int((pnl >= 0).sum())

    positives = np.maximum(pnl, 0.0)
    total_positive = float(positives.sum())

    concentration = (
        float(positives.max() / total_positive)
        if total_positive > 0
        else 1.0
    )

    return nonnegative, concentration


def main():
    data = download_data()
    features = build_features(data)

    end = max(df.index.max() for df in features.values())
    oos_start = (end - pd.DateOffset(months=OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    dev_trades, dev_eq = simulate(
        "ROTATION_Development",
        features,
        dev_start,
        dev_end,
    )

    oos_trades, oos_eq = simulate(
        "ROTATION_OOS",
        features,
        oos_start,
        end,
    )

    dev = metrics(dev_trades, dev_eq)
    oos = metrics(oos_trades, oos_eq)

    dev_yearly = yearly_stats(dev_trades)
    evaluable = dev_yearly[dev_yearly["trades"] >= 12]

    profitable_year_ratio = (
        float((evaluable["net_pnl"] > 0).mean())
        if len(evaluable)
        else 0.0
    )

    dev_symbols = per_symbol(dev_trades)
    oos_symbols = per_symbol(oos_trades)

    dev_nonnegative, dev_concentration = concentration_and_breadth(dev_symbols)

    development_pass = bool(
        dev["trades"] >= 250
        and dev["return_pct"] > 0
        and dev["win_rate_pct"] >= 52.0
        and dev["profit_factor"] >= 1.20
        and dev["expectancy"] > 0
        and dev["max_drawdown_pct"] <= 15.0
        and profitable_year_ratio >= 0.625
        and dev_nonnegative >= 8
        and dev_concentration <= 0.35
    )

    oos_diagnostic_pass = bool(
        oos["trades"] >= 50
        and oos["return_pct"] > 0
        and oos["win_rate_pct"] >= 52.0
        and oos["profit_factor"] >= 1.25
        and oos["expectancy"] > 0
        and oos["max_drawdown_pct"] <= 12.0
    )

    eligible = development_pass and oos_diagnostic_pass

    summary = {
        "strategy": "METALS_STOCKS_MONTHLY_RELATIVE_STRENGTH_ROTATION_FIXED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
            **dev,
            "profitable_year_ratio": round(profitable_year_ratio, 3),
            "evaluable_years": int(len(evaluable)),
            "nonnegative_symbols": dev_nonnegative,
            "positive_pnl_concentration": round(dev_concentration, 4),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
            **oos,
        },
        "development_pass": development_pass,
        "oos_diagnostic_pass": oos_diagnostic_pass,
        "eligible_for_fresh_forward_paper": eligible,
        "decision": (
            "ELIGIBLE: freeze rules and start fresh forward paper"
            if eligible
            else "FAIL: do not advance this rotation family"
        ),
        "real_money_status": "NOT APPROVED",
        "rules": {
            "rebalance_every_trading_days": REBALANCE_EVERY,
            "top_n": TOP_N,
            "fast_momentum_days": LOOKBACK_FAST,
            "slow_momentum_days": LOOKBACK_SLOW,
            "trend_sma_days": SMA_TREND,
            "score": "0.5*ret63 + 0.5*ret126",
            "market_regime": "SPY close > SMA200",
            "asset_filter": "close > SMA200 and ret63 > 0 and ret126 > 0",
            "fee_rate_per_side": FEE_RATE,
            "slippage_rate_per_side": SLIPPAGE_RATE,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"period": "Development", **dev},
        {"period": "OOS_diagnostic", **oos},
    ]).to_csv(OUT_COMPARISON, index=False)

    oos_symbols.to_csv(OUT_SYMBOLS, index=False)
    dev_yearly.to_csv(OUT_YEARLY, index=False)

    pd.DataFrame(
        [asdict(t) for t in dev_trades + oos_trades]
    ).to_csv(OUT_TRADES, index=False)

    oos_eq.to_csv(OUT_EQUITY, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
