# main.py

import asyncio
import json
import os
import time
import threading
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field

import requests
import websockets

from config import (
    TELEGRAM_TOKEN,
    TELEGRAM_ADMIN_ID,
    TELEGRAM_ADMIN_USERNAME,
    BINANCE_REST_URL,
    BINANCE_STREAM_URL,
    MIN_VOLUME_USDT,
    MAX_USDT_PAIRS,
    MIN_TIER_TO_SEND,
    SIGNAL_COOLDOWN_SECONDS,
)
from smc_logic import analyse_symbol
    # pastikan file smc_logic.py ada
from smc_scoring import score_smc_signal, tier_from_score, should_send_tier

# ===== FILE DATA PERSISTENT =====
SUBSCRIBERS_FILE = "subscribers.json"
VIP_FILE = "vip_users.json"
STATE_FILE = "bot_state.json"


# ================== GLOBAL STATE ==================

@dataclass
class BotState:
    scanning: bool = False
    running: bool = True
    last_update_id: Optional[int] = None

    last_signal_time: Dict[str, float] = field(default_factory=dict)
    min_tier: str = MIN_TIER_TO_SEND
    cooldown_seconds: int = SIGNAL_COOLDOWN_SECONDS
    debug: bool = False

    subscribers: Set[int] = field(default_factory=set)

    vip_users: Dict[int, float] = field(default_factory=dict)
    daily_counts: Dict[int, int] = field(default_factory=dict)
    daily_date: str = ""


state = BotState()


# ================== UTIL & STORAGE ==================

def is_admin(chat_id: int) -> bool:
    return TELEGRAM_ADMIN_ID and str(chat_id) == str(TELEGRAM_ADMIN_ID)


def load_subscribers() -> Set[int]:
    if not os.path.exists(SUBSCRIBERS_FILE):
        return set()
    try:
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(int(x) for x in data)
    except Exception as e:
        print("Gagal load subscribers:", e)
        return set()


def save_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(state.subscribers), f)
    except Exception as e:
        print("Gagal simpan subscribers:", e)


def load_vip_users() -> Dict[int, float]:
    if not os.path.exists(VIP_FILE):
        return {}
    try:
        with open(VIP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): float(v) for k, v in data.items()}
    except Exception as e:
        print("Gagal load VIP:", e)
        return {}


def save_vip_users():
    try:
        with open(VIP_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in state.vip_users.items()}, f)
    except Exception as e:
        print("Gagal simpan VIP:", e)


def is_vip(user_id: int) -> bool:
    """VIP jika expiry_ts > sekarang, atau jika dia admin."""
    now = time.time()
    if TELEGRAM_ADMIN_ID and str(user_id) == str(TELEGRAM_ADMIN_ID):
        return True
    exp = state.vip_users.get(user_id)
    return bool(exp and exp > now)


def cleanup_expired_vip():
    """Hapus VIP yang sudah kedaluwarsa dari memori + file."""
    now = time.time()
    expired_ids = [uid for uid, exp in state.vip_users.items() if exp <= now]
    if not expired_ids:
        return
    for uid in expired_ids:
        del state.vip_users[uid]
    save_vip_users()
    print("VIP expired dihapus otomatis:", expired_ids)


def load_bot_state():
    """Load scanning/min_tier/cooldown dari file (jika ada)."""
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.scanning = bool(data.get("scanning", False))
        state.min_tier = data.get("min_tier", state.min_tier)
        state.cooldown_seconds = int(data.get("cooldown_seconds", state.cooldown_seconds))
        print(
            f"Bot state loaded: scanning={state.scanning}, "
            f"min_tier={state.min_tier}, cooldown={state.cooldown_seconds}"
        )
    except Exception as e:
        print("Gagal load bot_state:", e)


def save_bot_state():
    """Simpan scanning/min_tier/cooldown ke file."""
    try:
        data = {
            "scanning": state.scanning,
            "min_tier": state.min_tier,
            "cooldown_seconds": state.cooldown_seconds,
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print("Gagal simpan bot_state:", e)


# ================== TELEGRAM ==================

def send_telegram(
    text: str,
    chat_id: Optional[int] = None,
    reply_markup: Optional[dict] = None,
) -> None:
    if not TELEGRAM_TOKEN:
        print("Telegram token belum di-set.")
        return

    if chat_id is None:
        if not TELEGRAM_ADMIN_ID:
            print("Tidak ada TELEGRAM_ADMIN_ID.")
            return
        chat_id = int(TELEGRAM_ADMIN_ID)

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if reply_markup is not None:
        data["reply_markup"] = json.dumps(reply_markup)

    try:
        r = requests.post(url, data=data, timeout=10)
        if not r.ok:
            print("Gagal kirim Telegram:", r.text)
    except Exception as e:
        print("Error kirim Telegram:", e)


def get_user_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "🔔 Aktifkan Sinyal", "callback_data": "user_activate"},
                {"text": "🔕 Nonaktifkan Sinyal", "callback_data": "user_deactivate"},
            ],
            [
                {"text": "📊 Status Saya", "callback_data": "user_status"},
                {"text": "⭐ Upgrade ke VIP", "callback_data": "user_upgrade"},
            ],
            [
                {"text": "❓ Bantuan", "callback_data": "user_help"},
            ],
        ]
    }


def get_admin_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "▶️ Mulai Scan", "callback_data": "admin_startscan"},
                {"text": "⏸️ Pause Scan", "callback_data": "admin_pausescan"},
            ],
            [
                {"text": "⚙️ Mode Tier", "callback_data": "admin_modetier"},
                {"text": "⏲️ Cooldown", "callback_data": "admin_cooldown"},
            ],
            [
                {"text": "📊 Status Bot", "callback_data": "admin_status"},
                {"text": "🔄 Restart Bot", "callback_data": "admin_restart"},
            ],
            [
                {"text": "⭐ VIP Control", "callback_data": "admin_vip"},
                {"text": "❓ Help Admin", "callback_data": "admin_help"},
            ],
        ]
    }


def broadcast_signal(text: str):
    """Kirim sinyal:
    - SELALU ke admin (unlimited)
    - Juga ke semua subscribers (FREE:max 2 sinyal per hari / VIP: unlimited)
    """
    # Reset harian & bersihkan VIP expired
    today = time.strftime("%Y-%m-%d")
    if state.daily_date != today:
        state.daily_date = today
        state.daily_counts = {}
        cleanup_expired_vip()
        print("Reset daily_counts & cleanup VIP untuk hari baru:", today)

    # 1) Selalu kirim ke ADMIN (kalau ada)
    if TELEGRAM_ADMIN_ID:
        try:
            send_telegram(text, chat_id=int(TELEGRAM_ADMIN_ID))
        except Exception as e:
            print("Gagal kirim ke admin:", e)
    else:
        print("⚠️ TELEGRAM_ADMIN_ID belum di-set. Admin tidak menerima sinyal.")

    # 2) Kirim ke subscribers (user)
    if not state.subscribers:
        print("Belum ada subscriber. Hanya admin yang menerima sinyal.")
        return

    for cid in list(state.subscribers):
        # Jangan proses admin lagi di sini (admin sudah dapat unlimited di atas)
        if TELEGRAM_ADMIN_ID and str(cid) == str(TELEGRAM_ADMIN_ID):
            continue

        # VIP → unlimited
        if is_vip(cid):
            send_telegram(text, chat_id=cid)
            continue

        # FREE → max 2 sinyal per hari
        count = state.daily_counts.get(cid, 0)
        if count >= 2:
            continue

        send_telegram(text, chat_id=cid)
        state.daily_counts[cid] = count + 1



# ================== BINANCE PAIRS ==================

def get_usdt_pairs(max_pairs: int) -> List[str]:
    """
    Ambil semua pair USDT yang statusnya TRADING,
    lalu filter hanya yang 24h quote volume >= MIN_VOLUME_USDT USDT.
    """
    # 1) Ambil info symbol (base/quote + status)
    info_url = f"{BINANCE_REST_URL}/api/v3/exchangeInfo"
    r = requests.get(info_url, timeout=10)
    r.raise_for_status()
    info = r.json()

    usdt_symbols = []
    for s in info["symbols"]:
        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
            usdt_symbols.append(s["symbol"])  # huruf besar

    # 2) Ambil 24h ticker semua symbol (sekali request)
    ticker_url = f"{BINANCE_REST_URL}/api/v3/ticker/24hr"
    r2 = requests.get(ticker_url, timeout=10)
    r2.raise_for_status()
    tickers = r2.json()  # list of dict

    vol_map: Dict[str, float] = {}
    for t in tickers:
        sym = t.get("symbol")
        if sym in usdt_symbols:
            # quoteVolume = volume dalam USDT
            try:
                qv = float(t.get("quoteVolume", "0"))
            except ValueError:
                qv = 0.0
            vol_map[sym] = qv

    # 3) Filter volume >= dalam USDT
    min_vol = MIN_VOLUME_USDT
    filtered = [s for s in usdt_symbols if vol_map.get(s, 0.0) >= min_vol]

    # 4) Urutkan dari volume terbesar → terkecil, lalu batasi max_pairs
    filtered_sorted = sorted(filtered, key=lambda s: vol_map.get(s, 0.0), reverse=True)

    # pakai huruf kecil seperti sebelumnya
    symbols_lower = [s.lower() for s in filtered_sorted]

    if max_pairs > 0:
        symbols_lower = symbols_lower[:max_pairs]

    print(f"Filter volume >= {min_vol:,.0f} USDT → {len(symbols_lower)} pair.")
    return symbols_lower



# ================== SIGNAL MESSAGE ==================

def build_signal_message(symbol: str, levels: dict, conditions: dict, score: int, tier: str) -> str:
    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]

    def yn(flag: bool) -> str:
        return "✅" if flag else "❌"

    bias_ok = conditions.get("bias_1h_not_bearish") or conditions.get("bias_1h_strong_bullish")
    disc_ok = conditions.get("in_discount_50_62") or conditions.get("in_discount_62_79")
    mb_or_bb = conditions.get("has_mitigation_block") or conditions.get("has_breaker_block")

    text = f"""
🟦 SMC–ICT INTRADAY SIGNAL — *{symbol}*

SMC SCORE: *{score}/150* — Tier *{tier}*

💰 *Harga*
• Entry : `{entry:.6f}`
• SL    : `{sl:.6f}`
• TP1   : `{tp1:.6f}`
• TP2   : `{tp2:.6f}`
• TP3   : `{tp3:.6f}`

📌 *Checklist SMC*
• Bias 1H      : {yn(bias_ok)}
• Struktur 15m : {yn(conditions.get("struct_15m_bullish"))}
• Sweep        : {yn(conditions.get("has_big_sweep"))}
• CHoCH        : {yn(conditions.get("has_choch_impulse"))}
• Discount     : {yn(disc_ok)}
• FVG          : {yn(conditions.get("has_fvg_fresh"))}
• MB / Breaker : {yn(mb_or_bb)}
• Liquidity    : {yn(conditions.get("liquidity_target_clear"))}

📝 *Catatan*
Free: maksimal 2 sinyal/hari. VIP: Unlimited sinyal.
"""
    return text


# ================== TELEGRAM COMMAND HANDLER ==================

def handle_user_start(chat_id: int):
    pkg = "VIP" if is_vip(chat_id) else "FREE"
    limit = "Unlimited" if is_vip(chat_id) else "2 sinyal per hari"
    active = "AKTIF" if chat_id in state.subscribers else "Tidak aktif"

    send_telegram(
        f"🟦 SMC INTRADAY SIGNAL BOT\n\n"
        f"Status Kamu:\n"
        f"• Paket : *{pkg}*\n"
        f"• Limit : *{limit}*\n"
        f"• Sinyal : *{active}*\n\n"
        f"Gunakan tombol di bawah untuk mengatur sinyal.",
        chat_id,
        reply_markup=get_user_keyboard(),
    )


def handle_admin_start(chat_id: int):
    send_telegram(
        "👑 *SMC INTRADAY — ADMIN PANEL*\n\n"
        "Bot siap. Gunakan menu di bawah untuk kontrol penuh.",
        chat_id,
        reply_markup=get_admin_keyboard(),
    )


def handle_command(cmd: str, args: list, chat_id: int):
    cmd = cmd.lower()

    # /start
    if cmd == "/start":
        if is_admin(chat_id):
            handle_admin_start(chat_id)
        else:
            handle_user_start(chat_id)
        return

    # /help → diarahkan ke tombol help
    if cmd == "/help":
        if is_admin(chat_id):
            send_telegram(
                "📖 Bantuan admin tersedia di tombol *❓ Help Admin* pada panel.",
                chat_id,
                reply_markup=get_admin_keyboard(),
            )
        else:
            send_telegram(
                "📖 Bantuan user tersedia di tombol *❓ Bantuan* pada menu.",
                chat_id,
                reply_markup=get_user_keyboard(),
            )
        return

    # ---------- USER COMMANDS ----------
    if not is_admin(chat_id):
        if cmd == "/activate":
            if chat_id in state.subscribers:
                send_telegram("ℹ️ Pencarian sinyal sudah *AKTIF*.", chat_id)
            else:
                state.subscribers.add(chat_id)
                save_subscribers()
                send_telegram("🔔 Pencarian sinyal *diaktifkan!*", chat_id)
            return

        if cmd == "/deactivate":
            if chat_id in state.subscribers:
                state.subscribers.remove(chat_id)
                save_subscribers()
                send_telegram("🔕 Pencarian sinyal *dinonaktifkan.*", chat_id)
            else:
                send_telegram("ℹ️ Pencarian sinyal sudah *tidak aktif*.", chat_id)
            return

        if cmd == "/mystatus":
            now = time.time()
            exp = state.vip_users.get(chat_id)
            if exp and exp > now:
                days_left = int((exp - now) / 86400)
                pkg = f"VIP (sisa ~{days_left} hari)"
                limit = "Unlimited"
            else:
                pkg = "FREE"
                limit = "2 sinyal per hari"

            active = "AKTIF ✅" if chat_id in state.subscribers else "TIDAK AKTIF ❌"
            send_telegram(
                "📊 *STATUS KAMU*\n\n"
                f"Paket  : *{pkg}*\n"
                f"Limit  : *{limit}*\n"
                f"Sinyal : *{active}*\n"
                f"User ID: `{chat_id}`",
                chat_id,
            )
            return

        # other user commands
        send_telegram("Perintah tidak dikenali. Gunakan tombol atau /start.", chat_id)
        return

    # ---------- ADMIN COMMANDS ----------
    # Admin zona
    if cmd == "/startscan":
        if state.scanning:
            send_telegram("ℹ️ Scan sudah *AKTIF*.", chat_id)
        else:
            state.scanning = True
            save_bot_state()
            send_telegram("▶️ Scan market *dimulai*.", chat_id)
        return

    if cmd == "/pausescan":
        if not state.scanning:
            send_telegram("ℹ️ Scan sudah *PAUSE*.", chat_id)
        else:
            state.scanning = False
            save_bot_state()
            send_telegram("⏸️ Scan market *dijeda*.", chat_id)
        return

    if cmd == "/status":
        send_telegram(
            "📊 *STATUS BOT*\n\n"
            f"Scan       : {'AKTIF' if state.scanning else 'STANDBY'}\n"
            f"Min Tier   : {state.min_tier}\n"
            f"Cooldown   : {state.cooldown_seconds} detik\n"
            f"Subscribers: {len(state.subscribers)} user\n"
            f"VIP Users  : {len(state.vip_users)} user\n",
            chat_id,
        )
        return

    if cmd == "/mode":
        if not args:
            send_telegram(
                "Mode sekarang:\n"
                f"- Min Tier: {state.min_tier}\n"
                "Gunakan: /mode aplus | a | b",
                chat_id,
            )
            return
        mode = args[0].lower()
        if mode == "aplus":
            state.min_tier = "A+"
        elif mode == "a":
            state.min_tier = "A"
        elif mode == "b":
            state.min_tier = "B"
        else:
            send_telegram("Mode tidak dikenali. Gunakan: aplus | a | b", chat_id)
            return
        save_bot_state()
        send_telegram(f"⚙️ Mode tier di-set ke: *{state.min_tier}*.", chat_id)
        return

    if cmd == "/cooldown":
        if not args:
            send_telegram(
                f"Cooldown sekarang: {state.cooldown_seconds} detik.\n"
                "Contoh: /cooldown 300  (5 menit)",
                chat_id,
            )
            return
        try:
            cd = int(args[0])
            if cd < 0:
                raise ValueError
            state.cooldown_seconds = cd
            save_bot_state()
            send_telegram(f"⏲️ Cooldown di-set ke {cd} detik.", chat_id)
        except ValueError:
            send_telegram("Format salah. Gunakan: /cooldown 300", chat_id)
        return

    if cmd == "/addvip":
        if not args:
            send_telegram("Gunakan: /addvip <user_id> [hari]", chat_id)
            return
        try:
            target_id = int(args[0])
            days = int(args[1]) if len(args) > 1 else 30
        except ValueError:
            send_telegram("Format salah. Contoh: /addvip 123456789 30", chat_id)
            return
        now = time.time()
        new_exp = now + days * 86400
        state.vip_users[target_id] = new_exp
        save_vip_users()
        send_telegram(f"⭐ VIP aktif untuk `{target_id}` selama {days} hari.", chat_id)
        send_telegram(
            f"🎉 VIP kamu diaktifkan selama {days} hari.\n"
            "Sinyal kamu sekarang *unlimited* per hari.",
            target_id,
        )
        return

    if cmd == "/removevip":
        if not args:
            send_telegram("Gunakan: /removevip <user_id>", chat_id)
            return
        try:
            target_id = int(args[0])
        except ValueError:
            send_telegram("Format salah. Contoh: /removevip 123456789", chat_id)
            return
        if target_id in state.vip_users:
            del state.vip_users[target_id]
            save_vip_users()
            send_telegram(f"VIP user `{target_id}` dihapus.", chat_id)
            send_telegram("VIP kamu telah dinonaktifkan. Kembali ke paket FREE.", target_id)
        else:
            send_telegram("User tersebut tidak terdaftar sebagai VIP.", chat_id)
        return

    if cmd == "/debug":
        if not args:
            send_telegram(f"Debug: {'ON' if state.debug else 'OFF'}", chat_id)
            return
        val = args[0].lower()
        if val == "on":
            state.debug = True
            send_telegram("Debug *ON*.", chat_id)
        elif val == "off":
            state.debug = False
            send_telegram("Debug *OFF*.", chat_id)
        else:
            send_telegram("Gunakan: /debug on | off", chat_id)
        return

    if cmd == "/stopbot":
        state.running = False
        send_telegram("⛔ Bot akan berhenti. Jalankan ulang main.py untuk start lagi.", chat_id)
        return

    send_telegram("Perintah admin tidak dikenali.", chat_id)


# ================== CALLBACK HANDLER ==================

def handle_callback(data_cb: str, from_id: int, chat_id_cq: int):
    # USER callbacks
    if data_cb == "user_activate":
        handle_command("/activate", [], chat_id_cq)
        return
    if data_cb == "user_deactivate":
        handle_command("/deactivate", [], chat_id_cq)
        return
    if data_cb == "user_status":
        handle_command("/mystatus", [], chat_id_cq)
        return
    if data_cb == "user_upgrade":
        send_telegram(
            "⭐ *UPGRADE KE VIP*\n\n"
            "Paket VIP memberikan:\n"
            "• Sinyal *unlimited* setiap hari\n"
            "• Fokus pada Tier tinggi\n"
            "• Masa aktif default 30 hari\n\n"
            "Hubungi admin untuk upgrade:\n"
            f"`{TELEGRAM_ADMIN_USERNAME}` (Forward pesan /mystatus kamu).",
            chat_id_cq,
        )
        return
    if data_cb == "user_help":
        send_telegram(
            "📖 *BANTUAN PENGGUNA*\n\n"
            "🔔 *Aktifkan Sinyal*\n"
            "• Menghidupkan pencarian sinyal.\n"
            "  FREE = 2 sinyal/hari, VIP = Unlimited.\n\n"
            "🔕 *Nonaktifkan Sinyal*\n"
            "• Mematikan pencarian sinyal. Tidak akan menerima sinyal.\n\n"
            "📊 *Status Saya*\n"
            "• Menampilkan status akun kamu (Free/VIP, limit harian, masa aktif).\n\n"
            "⭐ *Upgrade ke VIP*\n"
            "• Informasi paket VIP dan cara meningkatkan keanggotaan.\n\n"
            "❓ *Bantuan*\n"
            "• Menampilkan panduan penggunaan bot.",
            chat_id_cq,
        )
        return

    # ADMIN callbacks
    if not is_admin(from_id):
        send_telegram("Tombol ini hanya untuk admin.", chat_id_cq)
        return

    if data_cb == "admin_startscan":
        handle_command("/startscan", [], chat_id_cq)
        return
    if data_cb == "admin_pausescan":
        handle_command("/pausescan", [], chat_id_cq)
        return
    if data_cb == "admin_status":
        handle_command("/status", [], chat_id_cq)
        return
    if data_cb == "admin_modetier":
        send_telegram(
            "⚙️ *Mode Tier*\n\n"
            "Gunakan command:\n"
            "`/mode aplus` — hanya Tier A+\n"
            "`/mode a`     — Tier A & A+\n"
            "`/mode b`     — Tier B, A, A+",
            chat_id_cq,
        )
        return
    if data_cb == "admin_cooldown":
        send_telegram(
            "⏲️ *Cooldown Sinyal*\n\n"
            "Atur jarak minimal antar sinyal per pair.\n"
            "Contoh:\n"
            "`/cooldown 300`  (5 menit)\n"
            "`/cooldown 900`  (15 menit)\n"
            "`/cooldown 1800` (30 menit)",
            chat_id_cq,
        )
        return
    if data_cb == "admin_restart":
        state.last_signal_time.clear()
        send_telegram("🔄 Bot logic di-reset (last_signal_time dibersihkan).", chat_id_cq)
        return
    if data_cb == "admin_vip":
        send_telegram(
            "⭐ *VIP CONTROL*\n\n"
            "Gunakan:\n"
            "`/addvip <user_id> [hari]` — aktifkan VIP\n"
            "`/removevip <user_id>` — hapus VIP user\n\n"
            "User ID bisa dilihat dari perintah /mystatus user.",
            chat_id_cq,
        )
        return
    if data_cb == "admin_help":
        send_telegram(
            "📖 *BANTUAN ADMIN*\n\n"
            "▶️ *Mulai Scan*\n"
            "• Mengaktifkan pemindaian market.\n\n"
            "⏸️ *Pause Scan*\n"
            "• Menghentikan pemindaian sementara.\n\n"
            "⚙️ *Mode Tier*\n"
            "• Mengatur kualitas sinyal (A+, A, B).\n\n"
            "⏲️ *Cooldown*\n"
            "• Jarak minimal antar sinyal per pair.\n\n"
            "📊 *Status Bot*\n"
            "• Menampilkan status bot & statistik.\n\n"
            "🔄 *Restart Bot*\n"
            "• Me-reset logika bot (tanpa mematikan proses).\n\n"
            "⭐ *VIP Control*\n"
            "• Mengelola VIP user.\n\n"
            "❓ *Help Admin*\n"
            "• Menampilkan panduan admin ini.",
            chat_id_cq,
        )
        return


def telegram_command_loop():
    if not TELEGRAM_TOKEN:
        print("Tidak ada TELEGRAM_TOKEN, command loop tidak dijalankan.")
        return

    print("Telegram command loop start...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"

    # sync awal: skip pesan lama
    try:
        r = requests.get(url, timeout=20)
        if r.ok:
            data = r.json()
            results = data.get("result", [])
            if results:
                state.last_update_id = results[-1]["update_id"]
                print(f"Sync Telegram: skip {len(results)} pesan lama.")
    except Exception as e:
        print("Error sync awal Telegram:", e)

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

                # pesan biasa
                msg = upd.get("message")
                if msg:
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    text = msg.get("text", "")

                    if not text or not text.startswith("/"):
                        continue

                    parts = text.strip().split()
                    cmd = parts[0]
                    args = parts[1:]

                    print(f"[TELEGRAM CMD] {chat_id} {cmd} {args}")
                    handle_command(cmd, args, chat_id)
                    continue

                # callback query
                cq = upd.get("callback_query")
                if cq:
                    callback_id = cq.get("id")
                    from_id = cq.get("from", {}).get("id")
                    data_cb = cq.get("data")
                    msg_cq = cq.get("message", {})
                    chat_cq = msg_cq.get("chat", {})
                    chat_id_cq = chat_cq.get("id")

                    print(f"[TELEGRAM CB] {from_id} {data_cb}")

                    # jawab callback biar loading di Telegram hilang
                    try:
                        answer_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                        requests.post(
                            answer_url,
                            data={"callback_query_id": callback_id},
                            timeout=10,
                        )
                    except Exception as e:
                        print("Error answerCallbackQuery:", e)

                    if data_cb:
                        handle_callback(data_cb, from_id, chat_id_cq)

        except Exception as e:
            print("Error di telegram_command_loop:", e)
            time.sleep(2)


# ================== MAIN SCAN LOOP (BINANCE WS) ==================

async def run_bot():
    # load data persistent
    state.subscribers = load_subscribers()
    state.vip_users = load_vip_users()
    state.daily_date = time.strftime("%Y-%m-%d")
    cleanup_expired_vip()
    load_bot_state()

    print(f"Loaded {len(state.subscribers)} subscribers, {len(state.vip_users)} VIP users.")

    print("Mengambil list USDT pairs...")
    symbols = get_usdt_pairs(MAX_USDT_PAIRS)
    print(f"Scan {len(symbols)} pair:", ", ".join(s.upper() for s in symbols))

    streams = "/".join([f"{s}@kline_5m" for s in symbols])
    ws_url = f"{BINANCE_STREAM_URL}?streams={streams}"

    # loop untuk reconnect stabil
    while state.running:
        try:
            print("Menghubungkan ke WebSocket...")
            async with websockets.connect(ws_url) as ws:
                print("WebSocket terhubung.")
                if state.scanning:
                    print("Scan sebelumnya AKTIF → melanjutkan scan otomatis.")
                else:
                    print("Bot dalam mode STANDBY. Gunakan /startscan untuk mulai scan.\n")

                while state.running:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    kline = data.get("data", {}).get("k", {})
                    if not kline:
                        continue

                    is_closed = kline.get("x", False)
                    symbol = kline.get("s", "").upper()

                    if not is_closed or not symbol:
                        continue

                    # jika scanning False → abaikan data, WS tetap hidup
                    if not state.scanning:
                        continue

                    now = time.time()
                    if state.cooldown_seconds > 0:
                        last_ts = state.last_signal_time.get(symbol)
                        if last_ts and now - last_ts < state.cooldown_seconds:
                            if state.debug:
                                print(f"[{symbol}] Skip cooldown ({int(now - last_ts)}s/{state.cooldown_seconds}s)")
                            continue

                    if state.debug:
                        print(f"[{time.strftime('%H:%M:%S')}] 5m close: {symbol}")

                    conditions, levels = analyse_symbol(symbol)
                    if not conditions or not levels:
                        continue

                    score = score_smc_signal(conditions)
                    tier = tier_from_score(score)

                    if not should_send_tier(tier, state.min_tier):
                        if state.debug:
                            print(f"[{symbol}] Tier {tier} < {state.min_tier}, skip.")
                        continue

                    text = build_signal_message(symbol, levels, conditions, score, tier)
                    broadcast_signal(text)

                    state.last_signal_time[symbol] = now
                    print(f"[{symbol}] Sinyal dikirim: Score {score}, Tier {tier}")

        except websockets.ConnectionClosed:
            print("WebSocket terputus. Reconnect dalam 5 detik...")
            await asyncio.sleep(5)
        except Exception as e:
            print("Error di run_bot (luar):", e)
            print("Coba reconnect dalam 5 detik...")
            await asyncio.sleep(5)

    print("run_bot selesai karena state.running = False")


if __name__ == "__main__":
    # Jalankan loop command Telegram di thread terpisah
    cmd_thread = threading.Thread(target=telegram_command_loop, daemon=True)
    cmd_thread.start()

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        state.running = False
        print("Bot dihentikan oleh user (CTRL+C).")
