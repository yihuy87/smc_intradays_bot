# smc_logic.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime

from config import BINANCE_REST_URL


# ================== DATA FETCHING ==================

def get_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Ambil data candlestick Binance (REST)."""
    url = f"{BINANCE_REST_URL}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ================== INDICATORS ==================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ============================================================
#               BALANCED MODE SMC DETECTOR BLOCKS
# ============================================================

# 1. 1H BIAS

def detect_bias_1h(df_1h: pd.DataFrame):
    close = df_1h["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    last = close.iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]
    e200 = ema200.iloc[-1]

    strong_bull = last > e20 > e50 > e200
    not_bear = last >= e50

    return bool(strong_bull), bool(not_bear)


# 2. 15m STRUCTURE

def detect_struct_15m_bullish(df_15m: pd.DataFrame) -> bool:
    highs = df_15m["high"].values
    lows = df_15m["low"].values

    if len(highs) < 6:
        return False

    return bool(highs[-1] > highs[-4] and lows[-1] > lows[-4])


# 3. SWEEP (5m)

def detect_sweep_5m(df_5m: pd.DataFrame, lookback: int = 8) -> bool:
    lows = df_5m["low"].values
    if len(lows) < lookback + 3:
        return False

    last_low = lows[-2]  # candle close sebelumnya
    prev_lows = lows[-lookback-2:-2]
    return bool(last_low < prev_lows.min())


# 4. CHoCH (5m)

def detect_choch_impulse_5m(df_5m: pd.DataFrame, lookback: int = 12) -> bool:
    highs = df_5m["high"].values
    opens = df_5m["open"].values
    closes = df_5m["close"].values

    if len(highs) < lookback + 2:
        return False

    last_close = closes[-1]
    last_open = opens[-1]

    prev_highs = highs[-lookback-2:-1]
    broke_high = last_close > prev_highs.max()

    body = abs(last_close - last_open)
    recent_bodies = np.abs(closes[-lookback:] - opens[-lookback:])
    impulsive = body > recent_bodies.mean() * 1.1

    return bool(broke_high and impulsive)


# 5. DISCOUNT

def detect_discount_zone_5m(df_5m: pd.DataFrame, window: int = 22):
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    if len(highs) < window:
        return False, False

    recent_high = highs[-window:].max()
    recent_low = lows[-window:].min()
    last_close = closes[-1]

    full_range = recent_high - recent_low
    if full_range <= 0:
        return False, False

    pos = (last_close - recent_low) / full_range

    in_50_62 = (0.50 <= pos <= 0.62)
    in_62_79 = (0.62 < pos <= 0.79)

    return bool(in_50_62), bool(in_62_79)


# 6. FVG

def detect_last_bullish_fvg(df_5m: pd.DataFrame, window: int = 35):
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    found = False
    fvg_low = fvg_high = 0.0

    start = max(2, len(df_5m) - window)
    for i in range(start, len(df_5m) - 1):
        if lows[i + 1] > highs[i]:
            found = True
            fvg_low = highs[i]
            fvg_high = lows[i + 1]

    if not found:
        return False, 0.0, 0.0

    last_close = closes[-1]
    if last_close > fvg_high * 1.01:
        return False, 0.0, 0.0

    return True, float(fvg_low), float(fvg_high)


# 7. Liquidity Target (15m)

def detect_liquidity_target_15m(df_15m: pd.DataFrame, window: int = 8, tolerance: float = 0.0025):
    highs = df_15m["high"].values

    if len(highs) < window:
        return False

    recent = highs[-window:]
    if recent.mean() == 0:
        return False

    return bool((recent.max() - recent.min()) / recent.mean() < tolerance)


# 8. Mitigation Block (5m)

def detect_mitigation_block_5m(df_5m: pd.DataFrame):
    opens = df_5m["open"].values
    closes = df_5m["close"].values
    highs = df_5m["high"].values

    start = max(0, len(opens) - 15)
    for i in range(start, len(opens) - 1):
        if closes[i] < opens[i]:  # bearish
            if highs[i + 1] > opens[i]:  # disapu bullish berikutnya
                mb_low = closes[i]
                mb_high = opens[i]
                return True, float(mb_low), float(mb_high)

    return False, 0.0, 0.0


# 9. Breaker Block (5m)

def detect_breaker_block_5m(df_5m: pd.DataFrame):
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    opens = df_5m["open"].values
    closes = df_5m["close"].values

    if len(highs) < 20:
        return False, 0.0, 0.0

    pivot_index = -8
    pivot_low = lows[pivot_index]

    window_after = lows[pivot_index:]
    if len(window_after) < 5:
        return False, 0.0, 0.0

    if lows[-1] > pivot_low and closes[-1] > max(highs[pivot_index:]):
        start = max(0, pivot_index - 5)
        for i in range(start, pivot_index):
            if closes[i] < opens[i]:
                bb_low = closes[i]
                bb_high = opens[i]
                return True, float(bb_low), float(bb_high)

    return False, 0.0, 0.0


# 10. Anti Fake Pump & Momentum Filter

def detect_anti_fake_pump_5m(df_5m: pd.DataFrame) -> bool:
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    if len(highs) < 25:
        return False

    ranges = highs - lows
    avg_range = ranges[-20:-2].mean()
    last_range = ranges[-1]

    if avg_range <= 0:
        return False

    if last_range > avg_range * 2.5:
        recent_high = highs[-25:].max()
        recent_low = lows[-25:].min()
        full = recent_high - recent_low
        if full <= 0:
            return True
        pos = (closes[-1] - recent_low) / full
        if pos > 0.8:
            return True

    return False


def detect_momentum_ok_5m(df_5m: pd.DataFrame) -> bool:
    closes = df_5m["close"]
    if len(closes) < 50:
        return True

    rsi_val = rsi(closes, 14).iloc[-1]
    macd_line, signal_line, hist = macd(closes)

    m_val = macd_line.iloc[-1]
    s_val = signal_line.iloc[-1]

    if rsi_val > 80 or rsi_val < 25:
        return False

    if m_val < s_val and hist.iloc[-1] < 0 and abs(hist.iloc[-1]) > abs(m_val) * 0.5:
        return False

    return True


# ================== ENTRY/SL/TP ==================

def build_entry_sl_tp_from_smc(df_5m: pd.DataFrame,
                               in_disc_50_62,
                               in_disc_62_79,
                               fvg_low, fvg_high,
                               mb_low, mb_high,
                               bb_low, bb_high):
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    recent_high = highs[-20:].max()
    recent_low = lows[-20:].min()
    last_close = closes[-1]

    full_range = recent_high - recent_low
    if full_range <= 0:
        full_range = max(1.0, abs(last_close) * 0.001)

    if in_disc_62_79:
        entry = recent_low + full_range * 0.68
    elif in_disc_50_62:
        entry = recent_low + full_range * 0.56
    else:
        entry = recent_low + full_range * 0.50

    if mb_low and mb_high:
        entry = (mb_low + mb_high) / 2.0
    elif bb_low and bb_high:
        entry = (bb_low + bb_high) / 2.0
    elif fvg_low and fvg_high:
        entry = (entry + fvg_low) / 2.0

    sl = recent_low - full_range * 0.15

    tp1 = last_close + full_range * 0.50
    tp2 = last_close + full_range * 1.00
    tp3 = last_close + full_range * 1.50

    return {
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
    }


# ================== ANALYZE SYMBOL ==================

def analyse_symbol(symbol: str):
    """
    Ambil data 1H, 15m, 5m → hitung kondisi SMC
    Return (conditions, levels) atau (None, None).
    """
    try:
        df_1h = get_klines(symbol, "1h", 200)
        df_15m = get_klines(symbol, "15m", 200)
        df_5m = get_klines(symbol, "5m", 200)
    except Exception as e:
        print(f"[{symbol}] ERROR fetching data:", e)
        return None, None

    bias_strong, bias_not_bear = detect_bias_1h(df_1h)
    struct_15m = detect_struct_15m_bullish(df_15m)

    sweep = detect_sweep_5m(df_5m)
    choch = detect_choch_impulse_5m(df_5m)
    in50, in62 = detect_discount_zone_5m(df_5m)

    fvg, fvg_low, fvg_high = detect_last_bullish_fvg(df_5m)
    liq = detect_liquidity_target_15m(df_15m)

    mb, mb_low, mb_high = detect_mitigation_block_5m(df_5m)
    bb, bb_low, bb_high = detect_breaker_block_5m(df_5m)

    fake_pump = detect_anti_fake_pump_5m(df_5m)
    momentum_ok = detect_momentum_ok_5m(df_5m)

    if fake_pump or not momentum_ok:
        return None, None

    conditions = {
        "bias_1h_strong_bullish": bias_strong,
        "bias_1h_not_bearish": bias_not_bear,
        "struct_15m_bullish": struct_15m,
        "has_big_sweep": sweep,
        "has_choch_impulse": choch,
        "in_discount_50_62": in50,
        "in_discount_62_79": in62,
        "has_fvg_fresh": fvg,
        "has_mitigation_block": mb,
        "has_breaker_block": bb,
        "liquidity_target_clear": liq,
    }

    levels = build_entry_sl_tp_from_smc(
        df_5m,
        in50, in62,
        fvg_low, fvg_high,
        mb_low, mb_high,
        bb_low, bb_high
    )

    return conditions, levels
