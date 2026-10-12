from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
import requests
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ====================== CONFIGURACIÓN ======================
import os
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = "1124722713"

PIP = 0.25
RR = 1.0

SESSIONS = {
    "Asia":   (time(20, 0), time(0, 0)),
    "London": (time(2, 0),  time(5, 0)),
    "NY_AM":  (time(7, 0),  time(11, 0)),
}

NY_TZ = pytz.timezone("America/New_York")

# ====================== TELEGRAM ======================
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

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
        return {"type": "bullish_sweep", "extreme": current["low"], "idx": len(df)-1}
    swing_high = recent["high"].max()
    if current["high"] > swing_high and current["close"] < swing_high:
        return {"type": "bearish_sweep", "extreme": current["high"], "idx": len(df)-1}
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
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "cisd_level": round(cisd_level, 2),
        "sweep_extreme": round(sweep["extreme"], 2),
        "time": str(df_nq.index[-1]),
        "risk_points": round(abs(risk), 2)
    }

# ====================== RUTAS FLASK ======================
@app.route("/", methods=["GET"])
def home():
    return "Bot NQ Señales activo"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.json
        print("Datos recibidos:", data)
        # Aquí más adelante procesaremos los datos de TradingView
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "error"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
