# Dashboard metals update
import json
import os
from flask import Flask, render_template_string

app = Flask(__name__)

STATE_FILE = os.getenv("STATE_FILE", "metals_state.json")

DEFAULT_STATE = {
    "equity": 1000.00,
    "cash": 1000.00,
    "daily_pnl": 0.00,
    "total_pnl": 0.00,
    "open_positions": [],
    "closed_trades": 0,
    "wins": 0,
    "losses": 0,
    "win_rate": 0.0,
    "max_drawdown": 0.0,
    "trades": []
}


def load_state():
    raw = None
    db_url = os.getenv("DATABASE_URL", "").strip()
    trades = []

    if db_url:
        try:
            import psycopg2

            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT state FROM metals_bot_state WHERE id = 1"
                    )
                    row = cur.fetchone()

                    if row:
                        raw = row[0]

                    try:
                        cur.execute(
                            """
                            SELECT action, symbol, price_eur, pnl_eur
                            FROM metals_trades
                            ORDER BY id DESC
                            LIMIT 10
                            """
                        )

                        trade_rows = cur.fetchall()
                        trade_rows.reverse()

                        trades = [
                            {
                                "action": r[0],
                                "symbol": r[1],
                                "price": round(float(r[2]), 2),
                                "pnl": round(float(r[3]), 2),
                            }
                            for r in trade_rows
                        ]

                    except Exception:
                        trades = []

        except Exception as exc:
            print(f"Dashboard database error: {exc}")
            

    if raw is None:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}

    positions = []
    unrealized_total = 0.0

    for symbol, p in raw.get("positions", {}).items():
        pnl = float(
            p.get(
                "unrealized_pnl",
                p.get("pnl", 0.0)
            ) or 0.0
        )

        unrealized_total += pnl

        entry = float(
            p.get(
                "entry_eur",
                p.get(
                    "entry_price_eur",
                    p.get("entry", 0)
                )
            ) or 0
        )

        current_price = float(
            p.get("current_price_eur", entry) or entry
        )

        stop = float(
            p.get("stop_eur", 0) or 0
        )

        take_profit = float(
            p.get("take_profit_eur", 0) or 0
        )

        positions.append({
            "symbol": symbol,
            "entry": round(entry, 2),
            "current": round(current_price, 2),
            "stop": round(stop, 2),
            "take_profit": round(take_profit, 2),
            "pnl": round(pnl, 2),
        })
        
                

    realized = float(raw.get("realized_pnl", 0.0) or 0.0)
    daily = float(raw.get("daily_pnl", 0.0) or 0.0)
    total_pnl = realized + unrealized_total

    closed = int(raw.get("closed_trades", 0) or 0)
    wins = int(raw.get("wins", 0) or 0)
    losses = int(raw.get("losses", 0) or 0)

    win_rate = (
        wins / closed * 100
        if closed > 0
        else 0.0
    )

    return {
        "equity": 1000.0 + total_pnl,
        "cash": float(raw.get("cash_eur", 1000.0) or 0.0),
        "daily_pnl": daily,
        "total_pnl": total_pnl,
        "open_positions": positions,
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": float(
            raw.get("max_drawdown", 0.0) or 0.0
        ),
        "trades": trades if trades else raw.get("trades", []),
    }


HTML = """
<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="20">
    <title>Mi Bot Metales</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #0b1118;
            color: #f2f5f7;
            font-family: Arial, Helvetica, sans-serif;
        }

        .container {
            max-width: 760px;
            margin: auto;
            padding: 28px 20px 50px;
        }

        h1 {
            font-size: 38px;
            margin: 0;
        }

        .subtitle {
            color: #8e9aa8;
            font-size: 20px;
            margin-top: 5px;
        }

        .online {
            float: right;
            background: #123a2b;
            color: #5ee0a0;
            padding: 14px 20px;
            border-radius: 30px;
            font-weight: bold;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 35px;
        }

        .card {
            background: #111922;
            border: 1px solid #293543;
            border-radius: 28px;
            padding: 25px;
            margin-top: 20px;
        }

        .label {
            color: #8e9aa8;
            font-size: 20px;
        }

        .big {
            font-size: 40px;
            font-weight: bold;
            margin-top: 12px;
        }

        .profit {
            color: #5ee0a0;
        }

        .loss {
            color: #ff7185;
        }

        .position {
            margin-top: 15px;
            padding: 15px;
            background: #0d141c;
            border-radius: 15px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            padding: 14px 5px;
            border-bottom: 1px solid #293543;
            text-align: left;
        }

        th {
            color: #8e9aa8;
        }

        .metals {
            line-height: 1.8;
        }

        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }

            .card {
                padding: 20px;
            }

            .big {
                font-size: 30px;
            }

            h1 {
                font-size: 32px;
            }
        }
    </style>
</head>

<body>
<div class="container">

    <div>
        <span class="online">● ONLINE</span>
        <h1>Mi Bot Metales</h1>
        <div class="subtitle">Paper trading · Oro, Plata y Metales</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Saldo total</div>
            <div class="big">{{ "%.2f"|format(s.equity) }} €</div>

            <div class="{{ 'profit' if s.total_pnl >= 0 else 'loss' }}">
                Resultado total: {{ "%+.2f"|format(s.total_pnl) }} €
            </div>
        </div>

        <div class="card">
            <div class="label">Efectivo</div>
            <div class="big">{{ "%.2f"|format(s.cash) }} €</div>

            <div class="{{ 'profit' if s.daily_pnl >= 0 else 'loss' }}">
                Hoy: {{ "%+.2f"|format(s.daily_pnl) }} €
            </div>
        </div>
    </div>

    <div class="card">
        <div class="label">Posiciones abiertas</div>

        {% if s.open_positions %}
            {% for p in s.open_positions %}
                <div class="position">
    <strong>{{ p.get("symbol", "-") }}</strong><br><br>

    Entrada:
    {{ "%.2f"|format(p.get("entry", 0)) }} €<br>

    Ahora:
    {{ "%.2f"|format(p.get("current", 0)) }} €<br>

    Stop Loss:
    {{ "%.2f"|format(p.get("stop", 0)) }} €<br>

    Take Profit:
    {{ "%.2f"|format(p.get("take_profit", 0)) }} €<br>

    PnL:
    <span class="{{ 'profit' if p.get('pnl', 0) >= 0 else 'loss' }}">
        {{ "%+.2f"|format(p.get("pnl", 0)) }} €
    </span>
</div>
            {% endfor %}
        {% else %}
            <p>Sin posiciones abiertas</p>
        {% endif %}
    </div>

    <div class="card">
        <div class="label">Rendimiento</div>

        <div style="font-size:24px; margin-top:15px;">
            Operaciones cerradas: <strong>{{ s.closed_trades }}</strong><br>
            Ganadas: <strong>{{ s.wins }}</strong> ·
            Perdidas: <strong>{{ s.losses }}</strong><br>
            Acierto: <strong>{{ "%.1f"|format(s.win_rate) }}%</strong><br>

            Beneficio realizado:
            <strong class="{{ 'profit' if s.total_pnl >= 0 else 'loss' }}">
                {{ "%+.2f"|format(s.total_pnl) }} €
            </strong><br>

            Drawdown máximo:
            <strong>{{ "%.2f"|format(s.max_drawdown) }} €</strong>
        </div>
    </div>

    <div class="card">
        <div class="label">Mercados vigilados</div>
        <div class="metals" style="font-size:21px; margin-top:12px;">
            🥇 XAU · Oro<br>
            🥈 XAG · Plata<br>
            🔶 Cobre<br>
            ⚪ Platino<br>
            ⚫ Paladio
        </div>
    </div>

    <div class="card">
        <div class="label">Últimas operaciones</div>

        {% if s.trades %}
        <table>
            <tr>
                <th>Acción</th>
                <th>Metal</th>
                <th>Precio</th>
                <th>PnL</th>
            </tr>

            {% for t in s.trades[-10:]|reverse %}
            <tr>
                <td>{{ t.get("action", "-") }}</td>
                <td>{{ t.get("symbol", "-") }}</td>
                <td>{{ t.get("price", "-") }}</td>
                <td class="{{ 'profit' if t.get('pnl',0) >= 0 else 'loss' }}">
                    {{ "%+.2f"|format(t.get("pnl",0)) }} €
                </td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
            <p>Todavía no hay operaciones registradas.</p>
        {% endif %}
    </div>

</div>
</body>
</html>
"""


@app.route("/")
def dashboard():
    state = load_state()
    return render_template_string(HTML, s=state)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
