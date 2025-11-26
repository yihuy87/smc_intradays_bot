# SMC ICT Intraday Bot (Signal Only, Mode B)

Bot ini:
- Scan beberapa pair **USDT** (default 50) via Binance
- Bekerja dengan **WebSocket 1m** (Mode B)
- Analisa multi-timeframe: **1H, 15m, 5m**
- Deteksi: bias 1H, struktur 15m, sweep + CHoCH + discount 5m, FVG & liquidity sederhana
- Hitung **SMC Score (0–120)** dan Tier: `B`, `A`, `A+`
- Hanya kirim sinyal ke Telegram jika Tier minimal `A` (bisa diatur)

> Bot ini **signal only**, tidak autotrade.

---

## Setup

### 1. Buat bot Telegram

- Chat ke **@BotFather**
- `/newbot` → ambil **BOT TOKEN**

### 2. Ambil chat ID

- Chat ke **@userinfobot**
- Catat `Your user ID` → itu **TELEGRAM_CHAT_ID**

### 3. Clone / download repo ini

```bash
git clone ...
cd smc-intraday-bot
