import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ==========================================================
# METALS PAPER TRADING BOT - SIMULACIÓN
# NO API KEYS - NO REAL ORDERS - NO REAL MONEY
# ==========================================================

STARTING_BALANCE_EUR = 1000.0

# Riesgo controlado durante la fase de pruebas
MAX_RISK_PER_TRADE_EUR = 10.0
MAX_OPEN_POSITIONS = 2
DAILY_LOSS_LIMIT_EUR = 40.0
RESERVE_EUR = 100.0

# Relación beneficio/riesgo
STOP_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_R = 2.0

# Esperamos varias velas antes de permitir salida por cambio de tendencia
MIN_HOLD_CANDLES = 3

INTERVAL = "5m"
RANGE = "5d"
POLL_SECONDS = 60


# ----------------------------------------------------------
# METALES
#
# De momento usamos precios públicos de Yahoo Finance.
# Más adelante conectaremos el mismo motor a Interactive
# Brokers Paper Trading.
#
# GC=F  = Oro
# SI=F  = Plata
# HG=F  = Cobre
# PL=F  = Platino
# PA=F  = Paladio
# ----------------------------------------------------------

MARKETS = {
    "XAU": {
        "name": "ORO",
        "ticker": "GC=F",
        "ibkr_symbol": "XAUUSD",
    },
    "XAG": {
        "name": "PLATA",
        "ticker": "SI=F",
        "ibkr_symbol": "XAGUSD",
    },
    "COPPER": {
        "name": "COBRE",
        "ticker": "HG=F",
        "ibkr_symbol": "HG",
    },
    "PLATINUM": {
        "name": "PLATINO",
        "ticker": "PL=F",
        "ibkr_symbol": "PL",
    },
    "PALLADIUM": {
        "name": "PALADIO",
        "ticker": "PA=F",
        "ibkr_symbol": "PA",
    },
}


DATA_DIR = Path("/data")

if not DATA_DIR.exists():
    DATA_DIR = Path(".")

STATE_FILE = DATA_DIR / "metals_paper_state.json"
TRADES_FILE = DATA_DIR / "metals_trades.csv"


# ==========================================================
# UTILIDADES
# ==========================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def utc_day():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def default_state():
    return {
        "cash_eur": STARTING_BALANCE_EUR,
        "positions": {},
        "realized_pnl": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_date": utc_day(),
        "equity_peak": STARTING_BALANCE_EUR,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
    }


def _database_url():
    import os
    return os.getenv("DATABASE_URL", "").strip()


def _ensure_state_table():
    db_url = _database_url()

    if not db_url:
        return False

    import psycopg2

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS metals_bot_state (
                    id INTEGER PRIMARY KEY,
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    return True


def save_state(state):
    db_url = _database_url()

    if db_url:
        try:
            import psycopg2

            _ensure_state_table()

            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO metals_bot_state (id, state, updated_at)
                        VALUES (1, %s::jsonb, NOW())
                        ON CONFLICT (id)
                        DO UPDATE SET
                            state = EXCLUDED.state,
                            updated_at = NOW()
                        """,
                        (json.dumps(state),),
                    )

            return

        except Exception as exc:
            print(f"Database save error: {exc}")

    STATE_FILE.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )


def load_state():
    db_url = _database_url()

    if db_url:
        try:
            import psycopg2

            _ensure_state_table()

            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT state FROM metals_bot_state WHERE id = 1"
                    )
                    row = cur.fetchone()

            if row:
                return row[0]

            state = default_state()

            if STATE_FILE.exists():
                state = json.loads(
                    STATE_FILE.read_text(encoding="utf-8")
                )

            save_state(state)
            return state

        except Exception as exc:
            print(f"Database load error: {exc}")

    if not STATE_FILE.exists():
        state = default_state()
        save_state(state)
        return state

    try:
        return json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        state = default_state()
        save_state(state)
        return state


def reset_daily_pnl(state):
    if state.get("daily_pnl_date") != utc_day():
        state["daily_pnl_date"] = utc_day()
        state["daily_pnl"] = 0.0


# ==========================================================
# DATOS DE MERCADO
# ==========================================================

def yahoo_chart(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    params = {
        "interval": INTERVAL,
        "range": RANGE,
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    result = payload["chart"]["result"][0]

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "time": timestamps,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )

    df = df.dropna()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    return df


def eurusd_rate():
    try:
        df = yahoo_chart("EURUSD=X")
        return float(df.iloc[-1]["close"])
    except Exception:
        return 1.15


def usd_to_eur(price_usd):
    rate = eurusd_rate()

    if rate <= 0:
        rate = 1.15

    return price_usd / rate


# ==========================================================
# INDICADORES
# ==========================================================

def add_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan"),
    )

    df["rsi14"] = 100 - (
        100 / (1 + rs)
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    return df.dropna()


# ==========================================================
# SEÑAL DE ENTRADA
# ==========================================================

def entry_signal(df):
    if len(df) < 60:
        return False, "Not enough candles", {}

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(current["close"])
    ema20 = float(current["ema20"])
    ema50 = float(current["ema50"])
    rsi = float(current["rsi14"])

    volume = float(current["volume"])
    volume_ma20 = float(current["volume_ma20"])

    atr = float(current["atr14"])

    if not all(
        math.isfinite(x)
        for x in [
            price,
            ema20,
            ema50,
            rsi,
            atr,
        ]
    ):
        return False, "Invalid market data", {}

    has_volume_data = (
    math.isfinite(volume)
    and math.isfinite(volume_ma20)
    and volume > 0
    and volume_ma20 > 0
)

    relative_volume = (
        volume / volume_ma20
        if has_volume_data
        else 1.0
    )

    trend = (
        price > ema20
        and ema20 > ema50
        and current["ema20"] >= previous["ema20"]
    )

    momentum = (
        rsi >= 48
        and rsi <= 72
        )

    
        
    volume_ok = (
    True
    if not has_volume_data
    else relative_volume >= 0.80
    )

    candle_ok = (
        current["close"]
        > current["open"]
    )

    confirmation = (
        current["close"]
        > previous["close"]
    )

    score = 0

    if trend:
        score += 35

    if momentum:
        score += 20

    score += 15 if has_volume_data and volume_ok else 0

    if candle_ok:
        score += 15

    if confirmation:
        score += 15

    metrics = {
        "score": score,
        "price": price,
        "rsi": rsi,
        "ema20": ema20,
        "ema50": ema50,
        "rvol": relative_volume,
        "atr": atr,
    }

    long_signal = (
        trend
        and score >= 55
    )

    if long_signal:
        return True, "LONG", metrics

    return False, "NO ENTRY", metrics


# ==========================================================
# TAMAÑO DE POSICIÓN
# ==========================================================

def calculate_quantity(
    cash_eur,
    entry_eur,
    stop_eur,
):
    risk_per_unit = abs(
        entry_eur - stop_eur
    )

    if risk_per_unit <= 0:
        return 0.0

    qty_by_risk = (
        MAX_RISK_PER_TRADE_EUR
        / risk_per_unit
    )

    available_cash = max(
        0,
        cash_eur - RESERVE_EUR,
    )

    qty_by_cash = (
        available_cash / entry_eur
        if entry_eur > 0
        else 0
    )

    return max(
        0,
        min(
            qty_by_risk,
            qty_by_cash,
        ),
    )


# ==========================================================
# REGISTRO
# ==========================================================

def log_trade(
    action,
    symbol,
    price_eur,
    qty,
    pnl=0.0,
    reason="",
):
    trade_time = now_iso()

    clean_reason = (
        reason
        .replace(",", ";")
        .replace("\n", " ")
    )

    # Guardar copia local CSV
    new_file = not TRADES_FILE.exists()

    with TRADES_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        if new_file:
            file.write(
                "time,action,symbol,"
                "price_eur,qty,pnl_eur,reason\n"
            )

        file.write(
            f"{trade_time},"
            f"{action},"
            f"{symbol},"
            f"{price_eur:.4f},"
            f"{qty:.8f},"
            f"{pnl:.2f},"
            f"{clean_reason}\n"
        )

    # Guardar también en PostgreSQL
    db_url = _database_url()

    if db_url:
        try:
            import psycopg2

            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS metals_trades (
                            id BIGSERIAL PRIMARY KEY,
                            trade_time TIMESTAMPTZ NOT NULL,
                            action TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            price_eur DOUBLE PRECISION NOT NULL,
                            qty DOUBLE PRECISION NOT NULL,
                            pnl_eur DOUBLE PRECISION NOT NULL DEFAULT 0,
                            reason TEXT
                        )
                        """
                    )

                    cur.execute(
                        """
                        INSERT INTO metals_trades (
                            trade_time,
                            action,
                            symbol,
                            price_eur,
                            qty,
                            pnl_eur,
                            reason
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            trade_time,
                            action,
                            symbol,
                            float(price_eur),
                            float(qty),
                            float(pnl),
                            clean_reason,
                        ),
                    )

        except Exception as exc:
            print(f"Database trade log error: {exc}")



# ==========================================================
# ABRIR POSICIÓN
# ==========================================================

def open_position(
    state,
    symbol,
    price_usd,
    atr_usd,
):
    if symbol in state["positions"]:
        return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    if state["daily_pnl"] <= -DAILY_LOSS_LIMIT_EUR:
        return

    entry_eur = usd_to_eur(price_usd)
    atr_eur = usd_to_eur(atr_usd)

    stop_eur = (
        entry_eur
        - atr_eur * STOP_ATR_MULTIPLIER
    )

    risk_distance = (
        entry_eur - stop_eur
    )

    take_profit_eur = (
        entry_eur
        + risk_distance * TAKE_PROFIT_R
    )

    qty = calculate_quantity(
        state["cash_eur"],
        entry_eur,
        stop_eur,
    )

    if qty <= 0:
        return

    cost = qty * entry_eur

    if cost > state["cash_eur"]:
        return

    state["cash_eur"] -= cost

    state["positions"][symbol] = {
        "entry_eur": entry_eur,
        "qty": qty,
        "stop_eur": stop_eur,
        "take_profit_eur": take_profit_eur,
        "opened_at": now_iso(),
        "candles_held": 0,
    }

    log_trade(
        "BUY_SIM",
        symbol,
        entry_eur,
        qty,
        0.0,
        "High confidence metals signal",
    )

    save_state(state)

    print(
        f"{symbol}: BUY_SIM | "
        f"Entry €{entry_eur:.2f} | "
        f"Stop €{stop_eur:.2f} | "
        f"TP €{take_profit_eur:.2f}"
    )


# ==========================================================
# CONTROL DE POSICIONES
# ==========================================================

def manage_position(
    state,
    symbol,
    df,
):
    position = state["positions"][symbol]

    current = df.iloc[-1]
    previous = df.iloc[-2]

    price_usd = float(
        current["close"]
    )

    price_eur = usd_to_eur(
        price_usd
    )

    position["candles_held"] = (
        position.get(
            "candles_held",
            0,
        )
        + 1
    )

    entry = position["entry_eur"]
    stop = position["stop_eur"]
    take_profit = position["take_profit_eur"]
    qty = position["qty"]

    pnl = (
        price_eur - entry
    ) * qty

    exit_reason = None

    if price_eur <= stop:
        exit_reason = "STOP LOSS"

    elif price_eur >= take_profit:
        exit_reason = "TAKE PROFIT"

    elif (
        position["candles_held"]
        >= MIN_HOLD_CANDLES
        and current["close"]
        < current["ema20"]
        and previous["close"]
        < previous["ema20"]
        and current["ema20"]
        < current["ema50"]
    ):
        exit_reason = (
            "Confirmed trend reversal"
        )

    if exit_reason is None:
        print(
            f"{symbol}: HOLD | "
            f"PnL €{pnl:.2f}"
        )
        return

    state["cash_eur"] += (
        qty * price_eur
    )

    state["realized_pnl"] += pnl
    state["daily_pnl"] += pnl

    state["closed_trades"] += 1

    if pnl > 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

    log_trade(
        "SELL_SIM",
        symbol,
        price_eur,
        qty,
        pnl,
        exit_reason,
    )

    del state["positions"][symbol]

    save_state(state)

    print(
        f"{symbol}: SELL_SIM | "
        f"PnL €{pnl:.2f} | "
        f"{exit_reason}"
    )


# ==========================================================
# EQUITY
# ==========================================================

def portfolio_equity(state):
    equity = state["cash_eur"]

    for symbol, position in state["positions"].items():

        try:
            ticker = MARKETS[symbol]["ticker"]

            df = yahoo_chart(ticker)

            price_usd = float(
                df.iloc[-1]["close"]
            )

            price_eur = usd_to_eur(
                price_usd
            )

            equity += (
                position["qty"]
                * price_eur
            )

        except Exception:
            equity += (
                position["qty"]
                * position["entry_eur"]
            )

    return equity


# ==========================================================
# CICLO PRINCIPAL
# ==========================================================

def run_cycle(state):
    reset_daily_pnl(state)

    equity = portfolio_equity(state)

    if equity > state.get(
        "equity_peak",
        STARTING_BALANCE_EUR,
    ):
        state["equity_peak"] = equity
    peak = state.get(
        "equity_peak",
        STARTING_BALANCE_EUR,
    )

    current_drawdown = max(
        0.0,
        peak - equity,
    )

    state["max_drawdown"] = max(
        state.get("max_drawdown", 0.0),
        current_drawdown,
    )

    state["max_drawdown_eur"] = state["max_drawdown"]
    print()
    print("=" * 65)
    print("METALS PAPER TRADING BOT - SIMULACIÓN")
    print(f"UTC: {now_iso()}")
    print(f"Virtual equity: €{equity:.2f}")
    print(f"Cash: €{state['cash_eur']:.2f}")
    print(
        f"Daily PnL: "
        f"€{state['daily_pnl']:.2f}"
    )
    print(
        f"Open positions: "
        f"{len(state['positions'])}"
        f"/{MAX_OPEN_POSITIONS}"
    )
    print("=" * 65)

    for symbol, market in MARKETS.items():

        try:
            df = yahoo_chart(
                market["ticker"]
            )

            df = add_indicators(df)

            if symbol in state["positions"]:

                manage_position(
                    state,
                    symbol,
                    df,
                )

                continue

            if (
                len(state["positions"])
                >= MAX_OPEN_POSITIONS
            ):
                print(
                    f"{symbol}: NO ENTRY | "
                    "Position limit"
                )
                continue

            if (
                state["daily_pnl"]
                <= -DAILY_LOSS_LIMIT_EUR
            ):
                print(
                    f"{symbol}: NO ENTRY | "
                    "Daily loss limit"
                )
                continue

            signal, reason, metrics = (
                entry_signal(df)
            )

            if metrics:

                print(
                    f"{symbol} "
                    f"{market['name']}: "
                    f"{reason} | "
                    f"SCORE={metrics['score']}/100 "
                    f"RSI={metrics['rsi']:.1f} "
                    f"RVOL={metrics['rvol']:.2f}"
                )

            else:
                print(
                    f"{symbol}: {reason}"
                )

            if signal:

                open_position(
                    state,
                    symbol,
                    metrics["price"],
                    metrics["atr"],
                )

        except Exception as error:

            print(
                f"{symbol}: DATA ERROR | "
                f"{error}"
            )

    save_state(state)


def main():
    print(
        "Starting METALS PAPER TRADING BOT"
    )
    print(
        "SIMULATION ONLY - NO REAL ORDERS"
    )

    state = load_state()

    while True:

        try:
            run_cycle(state)

        except Exception as error:
            print(
                f"MAIN LOOP ERROR: {error}"
            )

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
