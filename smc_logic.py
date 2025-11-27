# smc_logic.py

import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime

from config import BINANCE_REST_URL


# ================== DATA FETCHING & UTIL ==================

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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI klasik."""
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
#               LOGIC SMC INTRADAY 1H–15m–5m
# ============================================================

# ================== 1. 1H BIAS (lebih ketat) ==================

def detect_bias_1h(df_1h: pd.DataFrame):
    """
    Strong bullish:
    - EMA20 > EMA50 > EMA200
    - close terakhir di atas EMA20
    Not bearish:
    - close terakhir di atas EMA50
    """

    close = df_1h["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    last = close.iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]
    e200 = ema200.iloc[-1]

    strong_bull = (last > e20 > e50 > e200)

    # not bearish: harga tidak di bawah EMA50
    not_bear = last >= e50

    return bool(strong_bull), bool(not_bear)


# ================== 2. 15m STRUCTURE ==================

def detect_struct_15m_bullish(df_15m: pd.DataFrame) -> bool:
    """
    Struktur 15m bullish sederhana:
    - perbandingan swing kecil: HH & HL
    - hindari kondisi jelas-jelas makin rendah (downtrend)
    """
    highs = df_15m["high"].values
    lows = df_15m["low"].values

    if len(highs) < 12:
        return False

    # pakai 3 swing kecil (jarak beberapa candle)
    h1 = highs[-4]
    h2 = highs[-2]
    l1 = lows[-4]
    l2 = lows[-2]

    hh_hl = (h2 > h1) and (l2 > l1)
    down = (h2 < h1) and (l2 < l1)

    if down:
        return False

    return bool(hh_hl)


# ================== 3. SWEEP 5m (lebih jelas) ==================

def detect_sweep_5m(df_5m: pd.DataFrame, lookback: int = 10) -> bool:
    """
    Sweep 5m:
    - low candle close sebelumnya menembus low beberapa candle sebelum
    - selisihnya tidak terlalu kecil (noise)
    """
    lows = df_5m["low"].values
    highs = df_5m["high"].values

    if len(lows) < lookback + 4:
        return False

    # gunakan candle sudah close: -2
    last_low = lows[-2]
    last_high = highs[-2]

    prev_lows = lows[-lookback-3:-3]
    base_min = prev_lows.min()

    if last_low >= base_min:
        return False

    # cek kedalaman sweep: minimal 0.2x range rata2
    ranges = highs[-lookback-3:-3] - lows[-lookback-3:-3]
    avg_range = ranges.mean() if len(ranges) > 0 else 0

    if avg_range <= 0:
        return False

    depth = base_min - last_low  # negatif kalau terbalik
    depth_abs = abs(depth)

    if depth_abs < avg_range * 0.2:
        # terlalu dangkal → noise
        return False

    # sweep oke
    return True


# ================== 4. CHoCH / Displacement 5m ==================

def detect_choch_impulse_5m(df_5m: pd.DataFrame, lookback: int = 14) -> bool:
    """
    Displacement / CHoCH bullish:
    - candle terakhir bullish
    - body besar dibanding rata2 body
    - close > high beberapa candle sebelumnya (break structure lokal)
    """
    highs = df_5m["high"].values
    opens = df_5m["open"].values
    closes = df_5m["close"].values
    lows = df_5m["low"].values

    if len(highs) < lookback + 3:
        return False

    last_open = opens[-1]
    last_close = closes[-1]
    last_high = highs[-1]
    last_low = lows[-1]

    # harus bullish
    if last_close <= last_open:
        return False

    prev_highs = highs[-lookback-2:-1]
    if last_close <= prev_highs.max():
        return False  # belum break

    # body & range check
    body = last_close - last_open
    all_bodies = np.abs(closes[-lookback-10:-1] - opens[-lookback-10:-1])
    avg_body = all_bodies.mean() if len(all_bodies) > 0 else 0

    if avg_body <= 0:
        return False

    # minimal 1.3x rata2 body
    if body < avg_body * 1.3:
        return False

    # wick atas tidak boleh dominan
    upper_wick = last_high - last_close
    total_range = last_high - last_low
    if total_range <= 0:
        return False

    if upper_wick / total_range > 0.4:
        # terlalu banyak rejection di atas
        return False

    return True


# ================== 5. DISCOUNT ZONE 5m ==================

def detect_discount_zone_5m(df_5m: pd.DataFrame, window: int = 30):
    """
    Hitung posisi harga terakhir dalam range swing 5m:
    pos = 0.0 (low) → 1.0 (high)

    Kita anggap:
    - 0.40–0.62 : discount moderat
    - 0.62–0.80 : deep discount ideal
    """
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

    in_50_62 = (0.40 <= pos <= 0.62)   # discount moderat
    in_62_79 = (0.62 < pos <= 0.80)    # deep discount

    return bool(in_50_62), bool(in_62_79)


# ================== 6. FVG 5m ==================

def detect_last_bullish_fvg(df_5m: pd.DataFrame, window: int = 40):
    """
    FVG bullish:
    - low candle n+1 > high candle n (gap ke atas)
    - FVG dianggap invalid jika harga sudah menutup penuh + 1% di atas
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    found = False
    fvg_low = fvg_high = 0.0

    start = max(2, len(df_5m) - window)
    for i in range(start, len(df_5m) - 1):
        # bullish FVG (gap ke atas)
        if lows[i + 1] > highs[i]:
            found = True
            fvg_low = highs[i]
            fvg_high = lows[i + 1]

    if not found:
        return False, 0.0, 0.0

    last_close = closes[-1]

    # invalid jika sudah benar2 ditembus jauh ke atas
    if last_close > fvg_high * 1.01:
        return False, 0.0, 0.0

    return True, float(fvg_low), float(fvg_high)


# ================== 7. Liquidity Cluster 15m ==================

def detect_liquidity_target_15m(df_15m: pd.DataFrame, window: int = 10, tolerance: float = 0.003):
    """
    Cari cluster high 15m sebagai target liquidity.
    """
    highs = df_15m["high"].values

    if len(highs) < window:
        return False

    recent = highs[-window:]
    if recent.mean() == 0:
        return False

    cluster = (recent.max() - recent.min()) / recent.mean() < tolerance
    return bool(cluster)


# ================== 8. Mitigation Block 5m ==================

def detect_mitigation_block_5m(df_5m: pd.DataFrame):
    """
    MB sederhana:
    - candle bearish → body (open→close) disapu oleh candle bullish berikutnya
    - area MB = body candle bearish tsb
    """
    opens = df_5m["open"].values
    closes = df_5m["close"].values
    highs = df_5m["high"].values

    start = max(0, len(opens) - 20)
    for i in range(start, len(opens) - 1):
        # bearish candle
        if closes[i] < opens[i]:
            # bullish setelahnya yang menembus open
            if closes[i + 1] > opens[i] and highs[i + 1] > opens[i]:
                mb_low = min(opens[i], closes[i])
                mb_high = max(opens[i], closes[i])
                return True, float(mb_low), float(mb_high)

    return False, 0.0, 0.0


# ================== 9. Breaker Block 5m ==================

def detect_breaker_block_5m(df_5m: pd.DataFrame):
    """
    Breaker sederhana:
    - cari pivot low relatif
    - jika kemudian harga membentuk higher low & break high,
      cari candle bearish sebelum pivot sebagai breaker.
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    opens = df_5m["open"].values
    closes = df_5m["close"].values

    if len(highs) < 30:
        return False, 0.0, 0.0

    # ambil pivot low sekitar -10
    pivot_index = -10
    pivot_low = lows[pivot_index]

    seg_lows = lows[pivot_index:]
    seg_highs = highs[pivot_index:]
    seg_closes = closes[pivot_index:]

    if len(seg_lows) < 6:
        return False, 0.0, 0.0

    # syarat: low terkini lebih tinggi dari pivot & close break high segmen
    if lows[-1] > pivot_low and closes[-1] > seg_highs.max():
        # cari bearish candle sebelum pivot
        start = max(0, pivot_index - 6)
        for i in range(start, pivot_index):
            if closes[i] < opens[i]:
                bb_low = min(opens[i], closes[i])
                bb_high = max(opens[i], closes[i])
                return True, float(bb_low), float(bb_high)

    return False, 0.0, 0.0


# ================== 10. Anti Fake Pump ==================

def detect_anti_fake_pump_5m(df_5m: pd.DataFrame) -> bool:
    """
    Deteksi pump ekstrem yang tidak layak dijadikan entry:
    - last candle range >> rata2
    - posisi harga dekat recent high (premium)
    """
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    if len(highs) < 30:
        return False

    ranges = highs - lows
    avg_range = ranges[-25:-3].mean()
    last_range = ranges[-1]

    if avg_range <= 0:
        return False

    # pump jika range > 3x rata2
    if last_range > avg_range * 3.0:
        # posisi dekat high?
        recent_high = highs[-30:].max()
        recent_low = lows[-30:].min()
        full = recent_high - recent_low
        if full <= 0:
            return True
        pos = (closes[-1] - recent_low) / full
        if pos > 0.82:
            return True

    return False


# ================== 11. Momentum & Choppy Filter ==================

def detect_momentum_ok_5m(df_5m: pd.DataFrame) -> bool:
    """
    Filter momentum:
    - RSI tidak overbought ekstrem
    - RSI tidak oversold sangat dalam (trend turun kuat)
    - MACD tidak sangat bearish
    """
    closes = df_5m["close"]
    if len(closes) < 60:
        return True  # jangan terlalu strict kalau data kurang

    rsi_val = rsi(closes, 14).iloc[-1]
    macd_line, signal_line, hist = macd(closes)

    m_val = macd_line.iloc[-1]
    s_val = signal_line.iloc[-1]
    h_val = hist.iloc[-1]

    # overbought sangat tinggi
    if rsi_val > 80:
        return False

    # oversold dalam → tren turun kuat
    if rsi_val < 22:
        return False

    # MACD sangat bearish
    if m_val < s_val and h_val < 0 and abs(h_val) > abs(m_val) * 0.7:
        return False

    return True


def detect_choppy_5m(df_5m: pd.DataFrame, window: int = 25) -> bool:
    """
    Deteksi market terlalu choppy / sempit:
    - range kecil
    - candle bergantian hijau-merah
    """
    closes = df_5m["close"].values
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    opens = df_5m["open"].values

    if len(closes) < window + 5:
        return False

    seg_high = highs[-window:]
    seg_low = lows[-window:]

    full_range = seg_high.max() - seg_low.min()
    avg_range = (seg_high - seg_low).mean()

    if avg_range <= 0:
        return True

    # jika full_range kecil (tidak jauh dari avg_range * 2) → choppy
    if full_range < avg_range * 2.5:
        return True

    # cek pergantian warna terlalu sering
    colors = np.sign(closes[-window:] - opens[-window:])
    # ubah 0 jadi 1
    colors[colors == 0] = 1
    flips = np.sum(colors[1:] != colors[:-1])

    if flips > window * 0.7:
        # 70% candle saling berganti warna → range chop
        return True

    return False


# ============================================================
#                  ENTRY / SL / TP GENERATION
# ============================================================

def build_entry_sl_tp_from_smc(df_5m: pd.DataFrame,
                               in_disc_50_62,
                               in_disc_62_79,
                               fvg_low, fvg_high,
                               mb_low, mb_high,
                               bb_low, bb_high):

    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values

    recent_high = highs[-30:].max()
    recent_low = lows[-30:].min()
    last_close = closes[-1]

    full_range = recent_high - recent_low
    if full_range <= 0:
        full_range = max(1.0, abs(last_close) * 0.001)

    # anchoring base di discount
    if in_disc_62_79:
        base_entry = recent_low + full_range * 0.68
    elif in_disc_50_62:
        base_entry = recent_low + full_range * 0.56
    else:
        base_entry = recent_low + full_range * 0.50

    entry = base_entry

    # confluence priority:
    # MB > Breaker > FVG > Discount
    if mb_low and mb_high and mb_high > mb_low:
        entry = (mb_low + mb_high) / 2.0
    elif bb_low and bb_high and bb_high > bb_low:
        entry = (bb_low + bb_high) / 2.0
    elif fvg_low and fvg_high and fvg_high > fvg_low:
        # entry di tengah FVG + mendekatkan ke base discount
        entry = (base_entry + (fvg_low + fvg_high) / 2.0) / 2.0

    # SL: sedikit di bawah recent_low
    sl = recent_low - full_range * 0.12

    # intraday TP (scalp intraday wajar)
    tp1 = last_close + full_range * 0.5   # target pendek
    tp2 = last_close + full_range * 1.0   # swing intraday
    tp3 = last_close + full_range * 1.5   # liquidity atas

    return {
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
    }


# ============================================================
#                    ANALYZE SYMBOL
# ============================================================

def analyse_symbol(symbol: str):
    """
    Analisa 1 symbol:
    - Ambil data 1H, 15m, 5m
    - Hitung kondisi SMC (bias, struktur, sweep, CHoCH, discount, FVG, MB, Breaker, liquidity)
    - Jalankan filter momentum, anti-pump, anti-choppy
    - Return (conditions, levels) atau (None, None) jika tidak layak
    """
    try:
        df_1h = get_klines(symbol, "1h", 200)
        df_15m = get_klines(symbol, "15m", 200)
        df_5m = get_klines(symbol, "5m", 220)
    except Exception as e:
        print(f"[{symbol}] ERROR fetching data:", e)
        return None, None

    # ===== Timeframe tinggi =====
    bias_strong, bias_not_bear = detect_bias_1h(df_1h)
    struct_15m = detect_struct_15m_bullish(df_15m)

    # ===== Trigger 5m =====
    sweep = detect_sweep_5m(df_5m)
    choch = detect_choch_impulse_5m(df_5m)
    in50, in62 = detect_discount_zone_5m(df_5m)

    fvg, fvg_low, fvg_high = detect_last_bullish_fvg(df_5m)
    liq = detect_liquidity_target_15m(df_15m)

    mb, mb_low, mb_high = detect_mitigation_block_5m(df_5m)
    bb, bb_low, bb_high = detect_breaker_block_5m(df_5m)

    fake_pump = detect_anti_fake_pump_5m(df_5m)
    momentum_ok = detect_momentum_ok_5m(df_5m)
    choppy = detect_choppy_5m(df_5m)

    # Filter keras:
    # - jika fake pump → skip
    # - jika momentum jelek → skip
    # - jika choppy parah → skip
    if fake_pump or (not momentum_ok) or choppy:
        return None, None

    # conditions untuk scoring
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

        "ema_alignment_bullish": bias_strong,
        "liquidity_target_clear": liq,
        "no_bearish_divergence": True,   # placeholder, bisa dikembangkan
        "no_exhaustion_sign": True,      # placeholder
    }

    levels = build_entry_sl_tp_from_smc(
        df_5m,
        in50, in62,
        fvg_low, fvg_high,
        mb_low, mb_high,
        bb_low, bb_high
    )

    return conditions, levels
