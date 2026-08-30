#!/usr/bin/env python3
"""
METALS & STOCKS — SINGLE PROFESSIONAL RESEARCH

One fixed strategy, no variants and no post-hoc optimization.

Trading universe:
FCX, SCCO, NEM, AEM, WPM, CCJ, BHP, RIO,
MSFT, GOOGL, AMZN, JPM

Benchmark/regime:
SPY

Logic:
- Long only.
- Close > EMA50 > EMA200.
- SPY above its EMA200.
- Close breaks prior 20-day high.
- 60-day relative strength vs SPY > 0.
- Volume >= 1.20 x 20-day average.
- ATR% between 1.2% and 9%.
- Entry next market open.
- Initial stop 2 ATR.
- Trailing stop 3 ATR.
- Max 60 trading days.
- Risk 0.5% equity, max EUR 5 planned risk.
- Max 4 positions, max 25% equity per position.
- No leverage.
- Fees + slippage included.

Validation:
- Data from 2018 onward.
- Last 18 months are locked OOS.
- Development and OOS each start with EUR 1,000.

OOS PASS:
- >= 25 closed trades
- net return > 0
- PF >= 1.20
- expectancy > 0
- max drawdown <= 18%
- >= 6 symbols with non-negative PnL
- no single symbol > 45% of positive PnL
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

TRADE_SYMBOLS = [
    "FCX", "SCCO", "NEM", "AEM", "WPM", "CCJ", "BHP", "RIO",
    "MSFT", "GOOGL", "AMZN", "JPM",
]
BENCHMARK = "SPY"

START_DATE = "2018-01-01"
OOS_MONTHS = 18

RISK_PCT = 0.005
MAX_RISK_EUR = 5.0
MAX_OPEN_POSITIONS = 4
MAX_POSITION_PCT = 0.25

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

BREAKOUT_DAYS = 20
RS_DAYS = 60
VOLUME_MULT = 1.20

EMA_FAST = 50
EMA_SLOW = 200
ATR_LEN = 14
STOP_ATR = 2.0
TRAIL_ATR = 3.0
MAX_HOLD_DAYS = 60

MIN_ATR_PCT = 0.012
MAX_ATR_PCT = 0.09

OUT_SUMMARY = Path("metals_stocks_summary.json")
OUT_TRADES = Path("metals_stocks_trades.csv")
OUT_SYMBOLS = Path("metals_stocks_oos_symbols.csv")
OUT_EQUITY = Path("metals_stocks_oos_equity.csv")
OUT_COMPARISON = Path("metals_stocks_comparison.csv")


def download_data():
    symbols = TRADE_SYMBOLS + [BENCHMARK]
    raw = yf.download(
        symbols,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No data downloaded from Yahoo Finance.")

    data = {}
    for sym in symbols:
        try:
            df = raw[sym].copy()
        except Exception:
            continue

        df.columns = [str(c).lower() for c in df.columns]
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            continue

        df = df[needed].dropna(subset=["open", "high", "low", "close"]).copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        if len(df) >= 300:
            data[sym] = df

    if BENCHMARK not in data:
        raise RuntimeError("SPY benchmark missing.")

    available = [s for s in TRADE_SYMBOLS if s in data]
    if len(available) < 8:
        raise RuntimeError(f"Too few trade symbols available: {available}")

    print("Available symbols:", available)
    return data


def prepare(df, spy):
    x = df.copy()

    x["ema50"] = x["close"].ewm(span=EMA_FAST, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    x["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()
    x["atr_pct"] = x["atr"] / x["close"]
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["prior_high20"] = x["high"].rolling(BREAKOUT_DAYS).max().shift(1)
    x["ret60"] = x["close"].pct_change(RS_DAYS)

    spy_close = spy["close"].reindex(x.index).ffill()
    spy_ema200 = spy_close.ewm(span=EMA_SLOW, adjust=False).mean()
    x["spy_regime"] = spy_close > spy_ema200
    x["spy_ret60"] = spy_close.pct_change(RS_DAYS)
    x["rs"] = x["ret60"] - x["spy_ret60"]

    return x.dropna().copy()


def signal_ok(row):
    return bool(
        row["close"] > row["ema50"] > row["ema200"]
        and row["close"] > row["prior_high20"]
        and row["rs"] > 0
        and row["vol_ma20"] > 0
        and row["volume"] >= VOLUME_MULT * row["vol_ma20"]
        and MIN_ATR_PCT <= row["atr_pct"] <= MAX_ATR_PCT
        and bool(row["spy_regime"])
    )


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    entry_date: str
    entry_fee: float
    stop: float
    atr_entry: float
    planned_risk: float
    highest_close: float
    days_held: int
    pending_exit: str | None


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
    planned_risk: float
    r_multiple: float
    hold_days: int
    exit_reason: str


def close_position(period, symbol, pos, exit_fill, exit_date, reason, cash, trades):
    exit_fee = pos.qty * exit_fill * FEE_RATE
    proceeds = pos.qty * exit_fill - exit_fee
    cash += proceeds

    entry_outlay = pos.qty * pos.entry + pos.entry_fee
    net = proceeds - entry_outlay

    trades.append(Trade(
        period=period,
        symbol=symbol,
        entry_date=pos.entry_date,
        exit_date=exit_date.strftime("%Y-%m-%d"),
        entry=pos.entry,
        exit=exit_fill,
        qty=pos.qty,
        net_pnl=net,
        return_pct=(exit_fill / pos.entry - 1.0) * 100.0,
        planned_risk=pos.planned_risk,
        r_multiple=(net / pos.planned_risk if pos.planned_risk > 0 else 0.0),
        hold_days=pos.days_held,
        exit_reason=reason,
    ))
    return cash


def simulate(period, prepared, start_dt, end_dt):
    all_dates = sorted(set().union(*[
        set(df.index[(df.index >= start_dt) & (df.index <= end_dt)])
        for df in prepared.values()
    ]))
    if not all_dates:
        return [], pd.DataFrame()

    cash = STARTING_CAPITAL
    positions = {}
    pending_entries = {}
    trades = []
    equity_rows = []
    peak = STARTING_CAPITAL

    for d in all_dates:
        # A) exits scheduled from yesterday's close -> today's open
        for sym in list(positions):
            pos = positions[sym]
            if not pos.pending_exit:
                continue
            df = prepared[sym]
            if d not in df.index:
                continue

            open_px = float(df.loc[d, "open"])
            exit_fill = open_px * (1.0 - SLIPPAGE_RATE)
            cash = close_position(
                period, sym, pos, exit_fill, d, pos.pending_exit, cash, trades
            )
            del positions[sym]

        # B) entries scheduled from yesterday's close -> today's open
        for sym in list(pending_entries):
            if sym in positions:
                del pending_entries[sym]
                continue
            if len(positions) >= MAX_OPEN_POSITIONS:
                del pending_entries[sym]
                continue

            df = prepared[sym]
            if d not in df.index:
                continue

            meta = pending_entries.pop(sym)
            open_px = float(df.loc[d, "open"])
            atr_signal = float(meta["atr"])

            # current equity before sizing
            equity = cash
            for psym, p in positions.items():
                pdf = prepared[psym]
                mark = float(pdf.loc[d, "open"]) if d in pdf.index else p.entry
                equity += p.qty * mark

            risk_per_share = STOP_ATR * atr_signal
            planned_risk = min(equity * RISK_PCT, MAX_RISK_EUR)

            qty_risk = planned_risk / risk_per_share if risk_per_share > 0 else 0.0
            qty_cap = (equity * MAX_POSITION_PCT) / open_px
            qty_cash = cash / (open_px * (1 + SLIPPAGE_RATE) * (1 + FEE_RATE))

            qty = min(qty_risk, qty_cap, qty_cash)
            if qty <= 0:
                continue

            entry_fill = open_px * (1.0 + SLIPPAGE_RATE)
            entry_fee = qty * entry_fill * FEE_RATE
            outlay = qty * entry_fill + entry_fee

            if outlay > cash:
                continue

            cash -= outlay
            positions[sym] = Position(
                symbol=sym,
                qty=qty,
                entry=entry_fill,
                entry_date=d.strftime("%Y-%m-%d"),
                entry_fee=entry_fee,
                stop=entry_fill - STOP_ATR * atr_signal,
                atr_entry=atr_signal,
                planned_risk=planned_risk,
                highest_close=entry_fill,
                days_held=0,
                pending_exit=None,
            )

        # C) manage positions with today's bar
        for sym in list(positions):
            pos = positions[sym]
            df = prepared[sym]
            if d not in df.index:
                continue
            row = df.loc[d]

            # Stop used here was known before today's bar.
            if float(row["low"]) <= pos.stop:
                exit_fill = pos.stop * (1.0 - SLIPPAGE_RATE)
                cash = close_position(
                    period, sym, pos, exit_fill, d, "STOP_OR_TRAIL", cash, trades
                )
                del positions[sym]
                continue

            pos.days_held += 1
            pos.highest_close = max(pos.highest_close, float(row["close"]))

            # New trailing stop is only for future bars.
            trail = pos.highest_close - TRAIL_ATR * float(row["atr"])
            pos.stop = max(pos.stop, trail)

            if float(row["close"]) < float(row["ema50"]):
                pos.pending_exit = "TREND_EXIT"
            elif pos.days_held >= MAX_HOLD_DAYS:
                pos.pending_exit = "TIME_EXIT"

        # D) end-of-day equity
        equity = cash
        for sym, pos in positions.items():
            df = prepared[sym]
            mark = float(df.loc[d, "close"]) if d in df.index else pos.entry
            equity += pos.qty * mark

        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        equity_rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "drawdown_pct": dd,
        })

        # E) generate new signals at today's close
        free_slots = MAX_OPEN_POSITIONS - len(positions) - len(pending_entries)
        if free_slots > 0:
            candidates = []

            for sym, df in prepared.items():
                if sym in positions or sym in pending_entries or d not in df.index:
                    continue
                row = df.loc[d]
                if signal_ok(row):
                    candidates.append((
                        sym,
                        float(row["rs"]),
                        float(row["volume"] / row["vol_ma20"]),
                        float(row["atr"]),
                    ))

            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            for sym, rs, rvol, atr in candidates[:free_slots]:
                pending_entries[sym] = {
                    "signal_date": d.strftime("%Y-%m-%d"),
                    "rs": rs,
                    "rvol": rvol,
                    "atr": atr,
                }

    # Force-close anything still open at final close.
    last_date = all_dates[-1]
    for sym in list(positions):
        pos = positions[sym]
        df = prepared[sym]
        avail = df.index[df.index <= last_date]
        if not len(avail):
            continue
        d = avail[-1]
        exit_fill = float(df.loc[d, "close"]) * (1.0 - SLIPPAGE_RATE)
        cash = close_position(
            period, sym, pos, exit_fill, d, "END_OF_TEST", cash, trades
        )
        del positions[sym]

    return trades, pd.DataFrame(equity_rows)


def metrics(trades, equity):
    if not trades:
        return {
            "trades": 0, "net_pnl": 0.0, "return_pct": 0.0,
            "win_rate_pct": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "max_drawdown_pct": 0.0,
            "avg_r": 0.0, "avg_hold_days": 0.0,
        }

    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
    gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "trades": int(len(trades)),
        "net_pnl": round(float(pnl.sum()), 2),
        "return_pct": round(float(pnl.sum() / STARTING_CAPITAL * 100), 3),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(pnl.mean()), 3),
        "max_drawdown_pct": round(float(equity["drawdown_pct"].max()) if len(equity) else 0.0, 3),
        "avg_r": round(float(np.mean([t.r_multiple for t in trades])), 3),
        "avg_hold_days": round(float(np.mean([t.hold_days for t in trades])), 2),
    }


def per_symbol(trades):
    rows = []
    for sym in TRADE_SYMBOLS:
        ts = [t for t in trades if t.symbol == sym]
        if not ts:
            rows.append({"symbol": sym, "trades": 0, "net_pnl": 0.0, "profit_factor": 0.0})
            continue

        pnl = np.array([t.net_pnl for t in ts], dtype=float)
        gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
        gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "symbol": sym,
            "trades": len(ts),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })
    return pd.DataFrame(rows)


def main():
    data = download_data()
    prepared = {
        s: prepare(data[s], data[BENCHMARK])
        for s in TRADE_SYMBOLS if s in data
    }

    common_end = max(df.index.max() for df in prepared.values())
    oos_start = (common_end - pd.DateOffset(months=OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    dev_trades, dev_eq = simulate("Development", prepared, dev_start, dev_end)
    oos_trades, oos_eq = simulate("OOS", prepared, oos_start, common_end)

    dev_m = metrics(dev_trades, dev_eq)
    oos_m = metrics(oos_trades, oos_eq)

    sym_df = per_symbol(oos_trades)
    nonnegative = int((sym_df["net_pnl"] >= 0).sum())

    positives = np.maximum(sym_df["net_pnl"].astype(float).values, 0.0)
    total_positive = float(positives.sum())
    concentration = float(positives.max() / total_positive) if total_positive > 0 else 1.0

    dev_pass = bool(
        dev_m["trades"] >= 60
        and dev_m["return_pct"] > 0
        and dev_m["profit_factor"] >= 1.10
        and dev_m["expectancy"] > 0
        and dev_m["max_drawdown_pct"] <= 25
    )

    oos_pass = bool(
        oos_m["trades"] >= 25
        and oos_m["return_pct"] > 0
        and oos_m["profit_factor"] >= 1.20
        and oos_m["expectancy"] > 0
        and oos_m["max_drawdown_pct"] <= 18
        and nonnegative >= 6
        and concentration <= 0.45
    )

    passed = dev_pass and oos_pass

    summary = {
        "strategy": "METALS_STOCKS_TREND_BREAKOUT_FIXED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {
            "start": str(dev_start.date()), "end": str(dev_end.date()), **dev_m
        },
        "oos": {
            "start": str(oos_start.date()), "end": str(common_end.date()), **oos_m
        },
        "oos_nonnegative_symbols": nonnegative,
        "oos_positive_pnl_concentration": round(concentration, 4),
        "development_pass": dev_pass,
        "oos_pass": oos_pass,
        "pass": passed,
        "decision": "PASS: build daily paper scanner" if passed else "FAIL: do not deploy",
        "universe": list(prepared.keys()),
        "benchmark": BENCHMARK,
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([asdict(t) for t in dev_trades + oos_trades]).to_csv(OUT_TRADES, index=False)
    sym_df.to_csv(OUT_SYMBOLS, index=False)
    oos_eq.to_csv(OUT_EQUITY, index=False)
    pd.DataFrame([
        {"period": "Development", **dev_m},
        {"period": "OOS", **oos_m},
    ]).to_csv(OUT_COMPARISON, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
