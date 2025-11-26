import asyncio
import json
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import threading

import websockets
import requests

from config import (
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    BINANCE_REST_URL,
    BINANCE_STREAM_URL,
    MAX_USDT_PAIRS,
    MIN_TIER_TO_SEND,
    SIGNAL_COOLDOWN_SECONDS,
)
from smc_logic import analyse_symbol
from smc_scoring import score_smc_signal, tier_from_score, should_send_tier


# ============ GLOBAL STATE ============

@dataclass
class BotState:
    scanning: bool = False          # False = mode siaga, True = scan & kirim sinyal
    running: bool = True            # False = stop bot
    last_update_id: Optional[int] = None  # untuk Telegram getUpdates offset
    last_signal_time: Dict[str, float] = field(default_factory=dict)  # pair -> timestamp terakhir

state = BotState()


# ============ TELEGRAM ============

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token / chat_id belum di-set. Lewati pengiriman.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if not r.ok:
            print("Gagal kirim Telegram:", r.text)
    except Exception as e:
        print("Error kirim Telegram:", e)


# ============ PAIRS ============

def get_usdt_pairs(max_pairs: int) -> List[str]:
    url = f"{BINANCE_REST_URL}/api/v3/exchangeInfo"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    symbols = []
    for s in data["symbols"]:
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
            symbols.append(s["symbol"].lower())

    symbols = sorted(symbols)
    return symbols[:max_pairs]


# ============ FORMAT PESAN ============

def build_signal_message(symbol: str, levels: dict, conditions: dict, score: int, tier: str) -> str:
    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]

    text = f"""
🟦 SMC–ICT INTRADAY SIGNAL — {symbol}

SMC SCORE: *{score}/150* — Tier *{tier}*

🎯 ENTRY TERBAIK
→ `{entry:.4f}`

🛡 STOP LOSS
→ `{sl:.4f}`

💰 TAKE PROFIT
→ TP1: `{tp1:.4f}`
→ TP2: `{tp2:.4f}`
→ TP3: `{tp3:.4f}`

📌 MODE & ENTRY MODE
- Bias 1H: {"strong bullish" if conditions.get("bias_1h_strong_bullish") else ("tidak bearish" if conditions.get("bias_1h_not_bearish") else "tidak mendukung")}
- Struktur 15m: {"bullish / CHoCH naik" if conditions.get("struct_15m_bullish") else "tidak bullish"}
- Trigger: {("Sweep + CHoCH + Discount" if (conditions.get("has_big_sweep") and conditions.get("has_choch_impulse") and (conditions.get("in_discount_50_62") or conditions.get("in_discount_62_79"))) else "Trigger belum lengkap")}

📌 Confluence
- FVG fresh: {conditions.get("has_fvg_fresh")}
- Mitigation Block: {conditions.get("has_mitigation_block")}
- Breaker Block: {conditions.get("has_breaker_block")}
- Liquidity target: {conditions.get("liquidity_target_clear")}

📌 Pump detector
Anti fake pump & momentum filter aktif.

📌 Catatan
Akurasi lebih tinggi jika EMA/BB/RSI/Wick/Volume terlihat.
"""
    return text


# ============ TELEGRAM COMMAND HANDLER ============

def handle_command(cmd: str, args: list):
    cmd = cmd.lower()

    # /start → hanya pesan sambutan, tidak mengubah state
    if cmd == "/start":
        send_telegram(
            "SMC INTRADAY siap ✅\n\n"
            "Ketik /startscan untuk *mulai scan & kirim sinyal*.\n"
            "Ketik /help untuk melihat *daftar command*."
        )

    elif cmd == "/startscan":
        if state.scanning:
            send_telegram("✅ Scan sudah *AKTIF*.")
        else:
            state.scanning = True
            send_telegram("▶ Scan *DIMULAI*.\nBot sekarang menganalisa market dan akan kirim sinyal.")

    elif cmd == "/pausescan":
        if not state.scanning:
            send_telegram("⏸ Scan sudah *PAUSE* / belum dimulai.")
        else:
            state.scanning = False
            send_telegram("⏸ Scan *DIPAUSE*.\nBot standby, tidak menganalisa & tidak kirim sinyal.")

    elif cmd == "/status":
        send_telegram(
            f"📊 STATUS BOT\n"
            f"- Scanning: {'AKTIF' if state.scanning else 'STANDBY'}\n"
            f"- Min Tier: {MIN_TIER_TO_SEND}\n"
            f"- Max USDT pairs: {MAX_USDT_PAIRS}"
        )

    elif cmd == "/help":
        send_telegram(
            "📖 *Command List*\n\n"
            "/startscan  - mulai scan & kirim sinyal\n"
            "/pausescan  - hentikan scan (standby)\n"
            "/status     - lihat status bot\n"
            "/help       - bantuan\n"
            "/stopbot    - hentikan bot (butuh run ulang di server)"
        )

    elif cmd == "/stopbot":
        state.running = False
        send_telegram("⛔ Bot akan berhenti. Jalankan lagi `python main.py` untuk start ulang.")

    else:
        send_telegram("Perintah tidak dikenali. Gunakan /help untuk daftar command.")



def telegram_command_loop():
    """Loop polling Telegram getUpdates untuk membaca command admin."""
    if not TELEGRAM_TOKEN:
        print("Tidak ada TELEGRAM_TOKEN, command loop tidak dijalankan.")
        return

    print("Telegram command loop start...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"

    # ==== SYNC AWAL: ABAIKAN SEMUA PESAN LAMA ====
    try:
        r = requests.get(url, timeout=20)
        if r.ok:
            data = r.json()
            results = data.get("result", [])
            if results:
                # set offset ke update_id terakhir, tapi TIDAK memproses isinya
                state.last_update_id = results[-1]["update_id"]
                print(f"Sync Telegram: skip {len(results)} pesan lama (last_update_id={state.last_update_id}).")
        else:
            print("Gagal sync awal Telegram:", r.text)
    except Exception as e:
        print("Error sync awal Telegram:", e)

    # ==== LOOP UTAMA: HANYA PROSES PESAN BARU ====
    while state.running:
        try:
            params = {}
            if state.last_update_id is not None:
                params["offset"] = state.last_update_id + 1

            r = requests.get(url, params=params, timeout=20)
            if not r.ok:
                print("Error getUpdates:", r.text)
                time.sleep(2)
                continue

            data = r.json()
            for upd in data.get("result", []):
                state.last_update_id = upd["update_id"]
                msg = upd.get("message")
                if not msg:
                    continue

                chat_id = msg.get("chat", {}).get("id")
                # Hanya izinkan command dari chat admin
                if str(chat_id) != str(TELEGRAM_CHAT_ID):
                    continue

                text = msg.get("text", "")
                if not text or not text.startswith("/"):
                    continue

                parts = text.strip().split()
                cmd = parts[0]
                args = parts[1:]

                print(f"[TELEGRAM CMD] {cmd} {args}")
                handle_command(cmd, args)

        except Exception as e:
            print("Error di telegram_command_loop:", e)
            time.sleep(2)



# ============ MAIN BOT LOOP (WEBSOCKET) ============

async def run_bot():
    print("Mengambil list USDT pairs...")
    symbols = get_usdt_pairs(MAX_USDT_PAIRS)
    print(f"Scan {len(symbols)} pair:", ", ".join(s.upper() for s in symbols))

    streams = "/".join([f"{s}@kline_1m" for s in symbols])
    ws_url = f"{BINANCE_STREAM_URL}?streams={streams}"

    print("Menghubungkan ke WebSocket...")
    async with websockets.connect(ws_url) as ws:
        print("Bot dalam mode *SIAGA* (standby).")
        print("Gunakan /startscan di Telegram untuk mulai scan.\n")

        while state.running:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                kline = data.get("data", {}).get("k", {})
                if not kline:
                    continue

                is_closed = kline.get("x", False)
                symbol = kline.get("s", "").upper()

                if not is_closed or not symbol:
                    continue

                # Jika scanning = False → standby, tidak analisa apa-apa
                if not state.scanning:
                    # mode siaga, lewati analisa
                    continue

                # ==== CEK COOLDOWN PER PAIR ====
                now = time.time()
                if SIGNAL_COOLDOWN_SECONDS > 0:
                    last_ts = state.last_signal_time.get(symbol)
                if last_ts and now - last_ts < SIGNAL_COOLDOWN_SECONDS:
                    # masih dalam cooldown → skip pair ini
                    continue    

                print(f"[{time.strftime('%H:%M:%S')}] 1m close: {symbol}")

                conditions, levels = analyse_symbol(symbol)
                if not conditions or not levels:
                    continue

                score = score_smc_signal(conditions)
                tier = tier_from_score(score)

                if not should_send_tier(tier, MIN_TIER_TO_SEND):
                    continue

                text = build_signal_message(symbol, levels, conditions, score, tier)
                send_telegram(text)

                # simpan waktu sinyal terakhir untuk pair ini
                state.last_signal_time[symbol] = now

                print(f"[{symbol}] Sinyal dikirim: Score {score}, Tier {tier}")

            except websockets.ConnectionClosed:
                print("WebSocket terputus. Reconnect dalam 5 detik...")
                await asyncio.sleep(5)
                return await run_bot()
            except Exception as e:
                print("Error di loop utama:", e)
                await asyncio.sleep(1)


if __name__ == "__main__":
    # Mulai thread untuk command Telegram
    cmd_thread = threading.Thread(target=telegram_command_loop, daemon=True)
    cmd_thread.start()

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        state.running = False
        print("Bot dihentikan oleh user.")
