import os
import warnings
from datetime import datetime, time
from typing import Dict, Optional

import pandas as pd
import pytz
import requests
import yfinance as yf
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ====================== CONFIGURACIÓN ======================
# El token y el chat se leen desde Render (Environment), no se escriben aquí.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1124722713")

# Clave opcional para que solo tu cron pueda llamar a /check (puedes dejarla sin crear)
CRON_SECRET = os.environ.get("CRON_SECRET")

PIP = 0.25
RR = 1.0

# Si la última vela tiene más de estos minutos de antigüedad, no se analiza
# (mercado cerrado, fin de semana o retraso de Yahoo)
MAX_DATA_AGE_MIN = 15

SESSIONS = {
    "Asia":   (time(20, 0), time(0, 0)),
    "London": (time(2, 0),  time(5, 0)),
    "NY_AM":  (time(7, 0),  time(11, 0)),
}

NY_TZ = pytz.timezone("America/New_York")

# Memoria para no enviar la misma señal dos veces
last_signal_key = None


# ====================== TELEGRAM ======================
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        print("Falta la variable TELEGRAM_BOT_TOKEN en Render")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"Telegram respondió {r.status_code}: {r.text}")
            return False
        return True
    except Exception as e:
        print(f"Error Telegram: {e}")
        return False


def format_signal_message(signal: Dict) -> str:
    emoji = "🟢 LONG" if signal["side"] == "LONG" else "🔴 SHORT"
    return (
        f"<b>{emoji} - NQ 3min</b>\n\n"
        f"⏰ {signal['time']}\n"
        f"📥 Entrada: <b>{signal['entry']}</b>\n"
        f"🛑 Stop Loss: <b>{signal['sl']}</b>\n"
        f"🎯 Take Profit: <b>{signal['tp']}</b>\n"
        f"📏 Riesgo: {signal['risk_points']} pts\n"
        f"CISD: {signal['cisd_level']}\n"
        f"Sweep: {signal['sweep_extreme']}"
    )


# ====================== DATOS DE MERCADO ======================
def download_3min(ticker: str) -> pd.DataFrame:
    """Descarga velas de 1 minuto de Yahoo y las convierte a velas de 3 minutos."""
    raw = yf.download(
        ticker,
        period="5d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        prepost=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo no devolvió datos para {ticker}")

    # Yahoo a veces devuelve columnas dobles; las dejamos simples
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close"]].dropna()

    # Hora de Nueva York
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(NY_TZ)

    last_1m = raw.index[-1]

    # Juntar de 3 en 3 minutos
    df3 = raw.resample("3min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    # Quitar la última vela si todavía no está completa
    # (una vela de 3 min que empieza en T termina en T+3min;
    #  está completa si ya tenemos la vela de 1 min de T+2min)
    if len(df3) > 0:
        last_bar_start = df3.index[-1]
        if last_1m < last_bar_start + pd.Timedelta(minutes=2):
            df3 = df3.iloc[:-1]

    return df3


# ====================== LÓGICA ESTRATEGIA ======================
def is_in_session(dt: datetime) -> bool:
    if dt.tzinfo is None:
        dt = NY_TZ.localize(dt)
    else:
        dt = dt.astimezone(NY_TZ)
    t = dt.time()
    for start, end in SESSIONS.values():
        if start < end:
            if start <= t <= end:
                return True
        else:
            if t >= start or t <= end:
                return True
    return False


def get_delivery_leg_open(df: pd.DataFrame, idx: int, direction: str) -> Optional[float]:
    if direction == "bullish":
        i = idx
        while i > 0 and df["close"].iloc[i] < df["open"].iloc[i]:
            i -= 1
        return df["open"].iloc[i + 1] if i + 1 <= idx else None
    else:
        i = idx
        while i > 0 and df["close"].iloc[i] > df["open"].iloc[i]:
            i -= 1
        return df["open"].iloc[i + 1] if i + 1 <= idx else None


def detect_sweep(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
    if len(df) < lookback + 5:
        return None
    recent = df.iloc[-lookback:]
    current = df.iloc[-1]
    swing_low = recent["low"].min()
    if current["low"] < swing_low and current["close"] > swing_low:
        return {"type": "bullish_sweep", "extreme": current["low"], "idx": len(df) - 1}
    swing_high = recent["high"].max()
    if current["high"] > swing_high and current["close"] < swing_high:
        return {"type": "bearish_sweep", "extreme": current["high"], "idx": len(df) - 1}
    return None


def detect_smt(df_nq: pd.DataFrame, df_es: pd.DataFrame, direction: str) -> bool:
    if len(df_nq) < 15 or len(df_es) < 15:
        return False
    nq_low = df_nq["low"].iloc[-5:].min()
    es_low = df_es["low"].iloc[-5:].min()
    nq_high = df_nq["high"].iloc[-5:].max()
    es_high = df_es["high"].iloc[-5:].max()
    prev_nq_low = df_nq["low"].iloc[-15:-5].min()
    prev_es_low = df_es["low"].iloc[-15:-5].min()
    prev_nq_high = df_nq["high"].iloc[-15:-5].max()
    prev_es_high = df_es["high"].iloc[-15:-5].max()
    if direction == "bullish":
        return (nq_low < prev_nq_low) and (es_low >= prev_es_low)
    else:
        return (nq_high > prev_nq_high) and (es_high <= prev_es_high)


def detect_cisd(df: pd.DataFrame, sweep: Dict) -> Optional[float]:
    direction = "bullish" if sweep["type"] == "bullish_sweep" else "bearish"
    cisd_level = get_delivery_leg_open(df, sweep["idx"], direction)
    if cisd_level is None:
        return None
    current_close = df["close"].iloc[-1]
    if direction == "bullish" and current_close > cisd_level:
        return cisd_level
    if direction == "bearish" and current_close < cisd_level:
        return cisd_level
    return None


def detect_cisd_failed(df: pd.DataFrame, cisd_level: float, direction: str) -> bool:
    if len(df) < 5:
        return False
    recent = df.iloc[-5:]
    if direction == "bullish":
        tried_below = any(recent["close"] < cisd_level)
        closed_back_above = df["close"].iloc[-1] > cisd_level
        return tried_below and closed_back_above
    else:
        tried_above = any(recent["close"] > cisd_level)
        closed_back_below = df["close"].iloc[-1] < cisd_level
        return tried_above and closed_back_below


def check_signal(df_nq: pd.DataFrame, df_es: pd.DataFrame) -> Optional[Dict]:
    if not is_in_session(df_nq.index[-1].to_pydatetime()):
        return None
    sweep = detect_sweep(df_nq)
    if not sweep:
        return None
    direction = "bullish" if sweep["type"] == "bullish_sweep" else "bearish"
    if not detect_smt(df_nq, df_es, direction):
        return None
    cisd_level = detect_cisd(df_nq, sweep)
    if cisd_level is None:
        return None
    if not detect_cisd_failed(df_nq, cisd_level, direction):
        return None
    entry = df_nq["close"].iloc[-1]
    if direction == "bullish":
        sl = sweep["extreme"] - PIP
        risk = entry - sl
        tp = entry + (risk * RR)
        side = "LONG"
    else:
        sl = sweep["extreme"] + PIP
        risk = sl - entry
        tp = entry - (risk * RR)
        side = "SHORT"
    return {
        "side": side,
        "entry": round(float(entry), 2),
        "sl": round(float(sl), 2),
        "tp": round(float(tp), 2),
        "cisd_level": round(float(cisd_level), 2),
        "sweep_extreme": round(float(sweep["extreme"]), 2),
        "time": str(df_nq.index[-1]),
        "risk_points": round(float(abs(risk)), 2),
    }


# ====================== RUTAS FLASK ======================
@app.route("/", methods=["GET"])
def home():
    return "Bot NQ Señales activo"


@app.route("/test", methods=["GET"])
def test():
    """Envía un mensaje de prueba a Telegram para comprobar que todo está conectado."""
    ok = send_telegram("✅ Prueba: el bot está conectado a Telegram.")
    return jsonify({"telegram_enviado": ok}), (200 if ok else 500)


@app.route("/check", methods=["GET"])
def check():
    """Revisa el mercado una vez. El cron la llama cada 3 minutos."""
    global last_signal_key

    if CRON_SECRET and request.args.get("key") != CRON_SECRET:
        return jsonify({"status": "no autorizado"}), 403

    try:
        df_nq = download_3min("NQ=F")
        df_es = download_3min("ES=F")

        # Usar solo las velas que existen en los dos mercados
        common = df_nq.index.intersection(df_es.index)
        df_nq = df_nq.loc[common]
        df_es = df_es.loc[common]

        if len(df_nq) < 25:
            return jsonify({"status": "pocos datos", "velas": len(df_nq)}), 200

        last_candle = df_nq.index[-1]
        age_min = (datetime.now(pytz.utc) - last_candle.astimezone(pytz.utc)).total_seconds() / 60
        if age_min > MAX_DATA_AGE_MIN:
            return jsonify({
                "status": "datos desactualizados (mercado cerrado o retraso)",
                "ultima_vela": str(last_candle),
                "minutos_de_antiguedad": round(age_min, 1),
            }), 200

        signal = check_signal(df_nq, df_es)
        if not signal:
            return jsonify({"status": "sin senal", "ultima_vela": str(last_candle)}), 200

        key = (signal["side"], signal["time"])
        if key == last_signal_key:
            return jsonify({"status": "senal repetida, no se reenvia", "senal": signal}), 200

        sent = send_telegram(format_signal_message(signal))
        if sent:
            last_signal_key = key
        return jsonify({"status": "senal enviada" if sent else "error al enviar a Telegram", "senal": signal}), 200

    except Exception as e:
        print("Error en /check:", e)
        return jsonify({"status": "error", "detalle": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
