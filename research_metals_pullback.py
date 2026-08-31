#!/usr/bin/env python3
"""
METALS & STOCKS — TREND PULLBACK RESEARCH

Purpose:
Test one new, pre-declared strategy family designed for a higher hit rate
than the previous breakout system. No parameter search, no variants.

Universe:
FCX, SCCO, NEM, AEM, WPM, CCJ, BHP, RIO,
MSFT, GOOGL, AMZN, JPM

Benchmark:
SPY

Entry (signal at close, execution next open):
- SPY close > SMA200
- Asset close > SMA200
- Asset EMA50 > EMA200
- RSI(2) <= 10
- Asset close < SMA5

Exit:
- Stop: 2.5 ATR, active from entry
- Mean-reversion exit: close > SMA5 -> next open
- Time exit: 10 trading days -> next open

Risk:
- 0.5% equity risk/trade, max EUR 5 planned risk
- max 4 open positions
- max 25% equity in one position
- no leverage
- fees + slippage included

Validation:
- 2018 onward
- last 18 months are OOS diagnostic
- Development determines whether the family is structurally credible
- Even if it passes, next step is fresh forward paper only

Development PASS:
- >= 100 trades
- net PnL > 0
- win rate >= 50%
- PF >= 1.20
- expectancy > 0
- max DD <= 12%
- profitable-year ratio >= 62.5%
- miners PF >= 1.05 and miner expectancy > 0

OOS diagnostic PASS:
- >= 25 trades
- net PnL > 0
- win rate >= 52%
- PF >= 1.30
- expectancy > 0
- max DD <= 10%
- miners net PnL > 0 and miners PF >= 1.20

No real-money approval is possible from this script.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research_metals_stocks as base


STARTING_CAPITAL = 1000.0
MINERS = {"FCX", "SCCO", "NEM", "AEM", "WPM", "CCJ", "BHP", "RIO"}

RISK_PCT = 0.005
MAX_RISK_EUR = 5.0
MAX_OPEN_POSITIONS = 4
MAX_POSITION_PCT = 0.25

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

EMA_FAST = 50
EMA_SLOW = 200
SMA_EXIT = 5
RSI_LEN = 2
RSI_ENTRY = 10.0
ATR_LEN = 14
STOP_ATR = 2.5
MAX_HOLD_DAYS = 10

OUT_SUMMARY = Path("metals_pullback_summary.json")
OUT_COMPARISON = Path("metals_pullback_comparison.csv")
OUT_SYMBOLS = Path("metals_pullback_oos_symbols.csv")
OUT_YEARLY = Path("metals_pullback_yearly.csv")
OUT_TRADES = Path("metals_pullback_trades.csv")
OUT_EQUITY = Path("metals_pullback_oos_equity.csv")


def rsi(series, length=2):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    avg_up = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # If avg_down is exactly zero, RSI should be 100.
    out = out.where(avg_down != 0.0, 100.0)
    return out


def prepare(df, spy):
    x = df.copy()

    x["ema50"] = x["close"].ewm(span=EMA_FAST, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    x["sma5"] = x["close"].rolling(SMA_EXIT).mean()
    x["rsi2"] = rsi(x["close"], RSI_LEN)

    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()

    spy_close = spy["close"].reindex(x.index).ffill()
    spy_sma200 = spy_close.rolling(EMA_SLOW).mean()
    x["spy_regime"] = spy_close > spy_sma200

    return x.dropna().copy()


def entry_signal(row):
    return bool(
        bool(row["spy_regime"])
        and row["close"] > row["ema200"]
        and row["ema50"] > row["ema200"]
        and row["rsi2"] <= RSI_ENTRY
        and row["close"] < row["sma5"]
    )


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    entry_date: str
    entry_fee: float
    stop: float
    planned_risk: float
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
        # 1) Execute exits scheduled from prior close at today's open.
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

        # 2) Execute entries scheduled from prior close at today's open.
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
                planned_risk=planned_risk,
                days_held=0,
                pending_exit=None,
            )

        # 3) Manage today's bar. Stop first for conservative execution.
        for sym in list(positions):
            pos = positions[sym]
            df = prepared[sym]

            if d not in df.index:
                continue

            row = df.loc[d]

            if float(row["low"]) <= pos.stop:
                exit_fill = pos.stop * (1.0 - SLIPPAGE_RATE)
                cash = close_position(
                    period, sym, pos, exit_fill, d, "STOP", cash, trades
                )
                del positions[sym]
                continue

            pos.days_held += 1

            if float(row["close"]) > float(row["sma5"]):
                pos.pending_exit = "SMA5_EXIT"
            elif pos.days_held >= MAX_HOLD_DAYS:
                pos.pending_exit = "TIME_EXIT"

        # 4) End-of-day equity.
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

        # 5) New signals at close for next open.
        free_slots = MAX_OPEN_POSITIONS - len(positions) - len(pending_entries)

        if free_slots > 0:
            candidates = []

            for sym, df in prepared.items():
                if sym in positions or sym in pending_entries or d not in df.index:
                    continue

                row = df.loc[d]

                if entry_signal(row):
                    # Most oversold first.
                    candidates.append((sym, float(row["rsi2"]), float(row["atr"])))

            candidates.sort(key=lambda z: z[1])

            for sym, rsi2_value, atr in candidates[:free_slots]:
                pending_entries[sym] = {
                    "signal_date": d.strftime("%Y-%m-%d"),
                    "rsi2": rsi2_value,
                    "atr": atr,
                }

    # Force close only for reporting. These will be identifiable.
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
            "trades": 0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_r": 0.0,
            "avg_hold_days": 0.0,
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
            float(equity["drawdown_pct"].max()) if len(equity) else 0.0, 3
        ),
        "avg_r": round(float(np.mean([t.r_multiple for t in trades])), 3),
        "avg_hold_days": round(float(np.mean([t.hold_days for t in trades])), 2),
    }


def subset_metrics(trades, symbols):
    ts = [t for t in trades if t.symbol in symbols]

    if not ts:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }

    pnl = np.array([t.net_pnl for t in ts], dtype=float)
    gp = float(pnl[pnl > 0].sum()) if np.any(pnl > 0) else 0.0
    gl = abs(float(pnl[pnl < 0].sum())) if np.any(pnl < 0) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "trades": int(len(ts)),
        "net_pnl": round(float(pnl.sum()), 2),
        "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(pnl.mean()), 3),
    }


def yearly_stats(trades):
    if not trades:
        return pd.DataFrame(columns=[
            "year", "trades", "net_pnl",
            "win_rate_pct", "profit_factor", "expectancy"
        ])

    df = pd.DataFrame([asdict(t) for t in trades])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    rows = []

    for year, g in df.groupby("year"):
        pnl = g["net_pnl"].astype(float)
        gp = float(pnl[pnl > 0].sum())
        gl = abs(float(pnl[pnl < 0].sum()))
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

        rows.append({
            "year": int(year),
            "trades": int(len(g)),
            "net_pnl": round(float(pnl.sum()), 2),
            "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def per_symbol(trades):
    rows = []

    for sym in base.TRADE_SYMBOLS:
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
            "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2),
            "profit_factor": round(float(pf), 3),
            "expectancy": round(float(pnl.mean()), 3),
        })

    return pd.DataFrame(rows)


def main():
    data = base.download_data()

    prepared = {
        s: prepare(data[s], data[base.BENCHMARK])
        for s in base.TRADE_SYMBOLS
        if s in data
    }

    end = max(df.index.max() for df in prepared.values())
    oos_start = (end - pd.DateOffset(months=base.OOS_MONTHS)).normalize()
    dev_start = pd.Timestamp(base.START_DATE)
    dev_end = oos_start - pd.Timedelta(days=1)

    dev_trades, dev_eq = simulate(
        "PULLBACK_Development", prepared, dev_start, dev_end
    )
    oos_trades, oos_eq = simulate(
        "PULLBACK_OOS", prepared, oos_start, end
    )

    dev = metrics(dev_trades, dev_eq)
    oos = metrics(oos_trades, oos_eq)

    dev_miners = subset_metrics(dev_trades, MINERS)
    oos_miners = subset_metrics(oos_trades, MINERS)

    y = yearly_stats(dev_trades)
    evaluable = y[y["trades"] >= 8].copy()

    profitable_year_ratio = (
        float((evaluable["net_pnl"] > 0).mean())
        if len(evaluable)
        else 0.0
    )

    development_pass = bool(
        dev["trades"] >= 100
        and dev["net_pnl"] > 0
        and dev["win_rate_pct"] >= 50.0
        and dev["profit_factor"] >= 1.20
        and dev["expectancy"] > 0
        and dev["max_drawdown_pct"] <= 12.0
        and profitable_year_ratio >= 0.625
        and dev_miners["profit_factor"] >= 1.05
        and dev_miners["expectancy"] > 0
    )

    oos_diagnostic_pass = bool(
        oos["trades"] >= 25
        and oos["net_pnl"] > 0
        and oos["win_rate_pct"] >= 52.0
        and oos["profit_factor"] >= 1.30
        and oos["expectancy"] > 0
        and oos["max_drawdown_pct"] <= 10.0
        and oos_miners["net_pnl"] > 0
        and oos_miners["profit_factor"] >= 1.20
    )

    eligible = development_pass and oos_diagnostic_pass

    summary = {
        "strategy": "METALS_STOCKS_TREND_PULLBACK_FIXED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {
            "start": str(dev_start.date()),
            "end": str(dev_end.date()),
            **dev,
            "miners": dev_miners,
            "profitable_year_ratio": round(profitable_year_ratio, 3),
            "evaluable_years": int(len(evaluable)),
        },
        "oos_diagnostic": {
            "start": str(oos_start.date()),
            "end": str(end.date()),
            **oos,
            "miners": oos_miners,
        },
        "development_pass": development_pass,
        "oos_diagnostic_pass": oos_diagnostic_pass,
        "eligible_for_fresh_forward_paper": eligible,
        "decision": (
            "ELIGIBLE: freeze parameters and start fresh forward paper"
            if eligible
            else "FAIL: do not advance this strategy"
        ),
        "parameters": {
            "rsi_len": RSI_LEN,
            "rsi_entry": RSI_ENTRY,
            "asset_trend": "close > EMA200 and EMA50 > EMA200",
            "market_regime": "SPY close > SMA200",
            "mean_reversion_exit": "close > SMA5, execute next open",
            "stop_atr": STOP_ATR,
            "max_hold_days": MAX_HOLD_DAYS,
            "risk_pct": RISK_PCT,
            "max_risk_eur": MAX_RISK_EUR,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_position_pct": MAX_POSITION_PCT,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
        },
        "real_money_status": "NOT APPROVED",
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"period": "Development", **dev},
        {"period": "OOS_diagnostic", **oos},
    ]).to_csv(OUT_COMPARISON, index=False)

    sym = per_symbol(oos_trades)
    sym.to_csv(OUT_SYMBOLS, index=False)

    y.to_csv(OUT_YEARLY, index=False)

    pd.DataFrame(
        [asdict(t) for t in dev_trades + oos_trades]
    ).to_csv(OUT_TRADES, index=False)

    oos_eq.to_csv(OUT_EQUITY, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
