# smc_scoring.py
"""
SMC Scoring v2 — Intraday 5m (lebih ketat)

Fokus:
- Bias 1H searah / minimal tidak bearish
- Struktur 15m tidak downtrend parah
- Sweep lokal 5m yang jelas
- Displacement / CHoCH yang benar-benar impuls
- Entry di area discount + FVG/MB/Breaker
- Ada tujuan liquidity di atas (target wajar)
"""

def score_smc_signal(c: dict) -> int:
    """
    Scoring SMC (0–150)
    """
    score = 0

    # ============================================================
    # 1. HIGHER TIMEFRAME (1H + 15m)
    # ============================================================
    if c.get("bias_1h_strong_bullish"):
        score += 30   # uptrend sehat
    elif c.get("bias_1h_not_bearish"):
        score += 15   # netral-bullish minimal

    if c.get("struct_15m_bullish"):
        score += 20   # struktur intraday naik

    # ============================================================
    # 2. TRIGGER 5m (Sweep + Displacement/CHoCH)
    # ============================================================
    if c.get("has_big_sweep"):
        score += 20

    if c.get("has_choch_impulse"):
        score += 25  # displacement sangat penting

    # ============================================================
    # 3. DISCOUNT ZONE 5m
    # ============================================================
    if c.get("in_discount_62_79"):
        score += 20  # zona paling ideal
    elif c.get("in_discount_50_62"):
        score += 12  # masih oke tapi kurang dalam

    # ============================================================
    # 4. FVG / IMBALANCE
    # ============================================================
    if c.get("has_fvg_fresh"):
        score += 20

    # ============================================================
    # 5. MB + BREAKER
    # ============================================================
    if c.get("has_mitigation_block"):
        score += 10

    if c.get("has_breaker_block"):
        score += 10

    # synergy MB + FVG
    if c.get("has_mitigation_block") and c.get("has_fvg_fresh"):
        score += 5

    # ============================================================
    # 6. LIQUIDITY TARGET
    # ============================================================
    if c.get("liquidity_target_clear"):
        score += 10

    # ============================================================
    # 7. SYNERGY BONUS: core SMC chain
    # ============================================================
    sweep = c.get("has_big_sweep")
    choch = c.get("has_choch_impulse")
    disc = c.get("in_discount_50_62") or c.get("in_discount_62_79")
    fvg = c.get("has_fvg_fresh")

    if sweep and choch and disc and fvg:
        score += 15  # setup lengkap core SMC

    return score


def tier_from_score(score: int) -> str:
    """
    Mapping score → Tier
    - A+ : >= 120
    - A  : 100–119
    - B  : 80–89
    - NONE : < 80
    """
    if score >= 120:
        return "A+"
    elif score >= 100:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "NONE"


def should_send_tier(tier: str, min_tier: str) -> bool:
    """
    Bandingkan tier terhadap min_tier:
    Urutan: NONE < B < A < A+
    """
    order = {"NONE": 0, "B": 1, "A": 2, "A+": 3}
    return order.get(tier, 0) >= order.get(min_tier, 2)
