"""
Inflow Nifty 500 Scanner module.
Compatible with main.py: from inflow_scanner import run_inflow_scanner
Daily yfinance data is historical daily OHLCV, not guaranteed intraday real-time.
"""
import datetime as dt
import logging
import math
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logger = logging.getLogger("inflow_scanner")
WATCHLIST_PATH = Path(__file__).resolve().with_name("swing_watchlist.json")
SL_PCT = 0.03
TARGET_PCT = 0.045


def _now_ist():
    return dt.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")


def load_swing_watchlist(path=None):
    """Load enabled watchlist entries and resolve configured Yahoo/NSE tickers.

    nse_symbol is the exchange symbol without the .NS suffix, for example
    RELIANCE. Entries without a confirmed symbol are reported, never guessed.
    """
    watchlist_path = Path(path) if path else WATCHLIST_PATH
    with watchlist_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("stocks", [])
    if not isinstance(entries, list):
        raise ValueError("Watchlist 'stocks' must be a list")

    configured, unconfigured = [], []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        name = str(entry.get("name", "")).strip()
        symbol = str(entry.get("nse_symbol") or "").strip().upper()
        if not name:
            unconfigured.append({"name": "(unnamed entry)", "reason": "Company name is missing"})
            continue
        if not symbol:
            unconfigured.append({"name": name, "reason": "nse_symbol is not set"})
            continue
        ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
        if ticker in seen:
            continue
        seen.add(ticker)
        configured.append({
            "name": name,
            "sector": str(entry.get("sector", "")),
            "symbol": symbol.removesuffix(".NS"),
            "ticker": ticker,
        })
    return configured, unconfigured


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    return result.mask((avg_loss == 0) & (avg_gain == 0), 50)


def _ticker_frame(data, ticker):
    """Extract ticker OHLCV regardless of yfinance MultiIndex orientation."""
    if data is None or data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()
    level0 = data.columns.get_level_values(0)
    level1 = data.columns.get_level_values(1)
    if ticker in level0:
        frame = data[ticker].copy()
    elif ticker in level1:
        frame = data.xs(ticker, axis=1, level=1).copy()
    else:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return frame


def run_inflow_scanner(capital_per_trade=30000.0):
    """Scan daily candles against the five agreed entry conditions."""
    try:
        capital = float(capital_per_trade)
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        return {"status": "error", "message": "Capital must be a positive number.", "data": []}

    try:
        watchlist, unconfigured = load_swing_watchlist()
    except Exception as exc:
        logger.exception("Swing watchlist could not be loaded")
        return {"status": "error", "message": "Could not load swing_watchlist.json: " + str(exc)[:200],
                "scanTime": _now_ist(), "totalMatches": 0, "data": []}

    tickers = [item["ticker"] for item in watchlist]
    ticker_info = {item["ticker"]: item for item in watchlist}
    diagnostics = {
        "tickerSource": "swing_watchlist.json",
        "watchlistEnabled": len(watchlist) + len(unconfigured),
        "tickersRequested": len(tickers),
        "unconfiguredStocks": unconfigured,
        "tickersWithData": 0,
        "tickersInsufficientHistory": 0,
        "tickersSkipped": 0,
        "tickerErrors": [],
        "dataInterval": "1d",
        "rules": [
            "Close > EMA200 OR EMA50 > EMA200",
            "Close within 2% of EMA20 OR EMA50",
            "Close > Open AND Close > previous Close",
            "RSI(14) between 38 and 68",
            "Volume >= 20-day average volume"
        ]
    }

    if not tickers:
        return {"status": "error",
                "message": "No enabled stocks have a configured nse_symbol in swing_watchlist.json. Add verified symbols to scan.",
                "scanTime": _now_ist(), "totalMatches": 0, "data": [], "diagnostics": diagnostics}

    try:
        data = yf.download(
            tickers=tickers, period="1y", interval="1d",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True, timeout=25
        )
    except Exception as exc:
        logger.exception("yfinance download failed")
        return {"status": "error", "message": "Market data download failed: " + str(exc)[:250],
                "scanTime": _now_ist(), "totalMatches": 0, "data": [], "diagnostics": diagnostics}

    if data is None or data.empty:
        return {"status": "error",
                "message": "Market data provider returned no data. Check Render logs/network.",
                "scanTime": _now_ist(), "totalMatches": 0, "data": [], "diagnostics": diagnostics}

    matches = []
    for ticker in tickers:
        try:
            df = _ticker_frame(data, ticker)
            if df.empty:
                diagnostics["tickersSkipped"] += 1
                continue
            required = {"Open", "Close", "Volume"}
            if not required.issubset(df.columns):
                diagnostics["tickersSkipped"] += 1
                if len(diagnostics["tickerErrors"]) < 30:
                    diagnostics["tickerErrors"].append(
                        {"ticker": ticker, "error": "Missing OHLCV columns"}
                    )
                continue

            df = df.dropna(subset=["Open", "Close", "Volume"]).copy()
            if len(df) < 200:
                diagnostics["tickersInsufficientHistory"] += 1
                continue
            diagnostics["tickersWithData"] += 1

            close = pd.to_numeric(df["Close"], errors="coerce")
            volume = pd.to_numeric(df["Volume"], errors="coerce")
            df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
            df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
            df["EMA_200"] = close.ewm(span=200, adjust=False).mean()
            df["RSI_14"] = calculate_rsi(close, 14)
            df["Vol_SMA20"] = volume.rolling(20, min_periods=20).mean()

            curr, prev = df.iloc[-1], df.iloc[-2]
            keys = ["Close", "Open", "EMA_20", "EMA_50", "EMA_200", "RSI_14", "Volume", "Vol_SMA20"]
            if any(pd.isna(curr[k]) for k in keys) or pd.isna(prev["Close"]):
                diagnostics["tickersSkipped"] += 1
                continue

            price = float(curr["Close"])
            ema20, ema50, ema200 = float(curr["EMA_20"]), float(curr["EMA_50"]), float(curr["EMA_200"])
            rsi, vol, vol_avg = float(curr["RSI_14"]), float(curr["Volume"]), float(curr["Vol_SMA20"])
            if price <= 0 or ema20 <= 0 or ema50 <= 0:
                diagnostics["tickersSkipped"] += 1
                continue

            c1 = price > ema200 or ema50 > ema200
            c2 = abs(price - ema20) / ema20 <= 0.02 or abs(price - ema50) / ema50 <= 0.02
            c3 = price > float(curr["Open"]) and price > float(prev["Close"])
            c4 = 38 <= rsi <= 68
            c5 = vol >= vol_avg

            if c1 and c2 and c3 and c4 and c5:
                qty = int(capital // price)
                if qty >= 1:
                    invested = round(qty * price, 2)
                    info = ticker_info.get(ticker, {})
                    matches.append({
                        "stock": ticker.removesuffix(".NS"),
                        "companyName": info.get("name", ticker.removesuffix(".NS")),
                        "sector": info.get("sector", ""),
                        "buyPrice": round(price, 2),
                        "qty": qty,
                        "investment": invested,
                        "stopLoss": round(price * (1 - SL_PCT), 2),
                        "target": round(price * (1 + TARGET_PCT), 2),
                        "rsi": round(rsi, 1),
                        "maxRisk": round(invested * SL_PCT, 2),
                        "targetProfit": round(invested * TARGET_PCT, 2)
                    })
        except Exception as exc:
            diagnostics["tickersSkipped"] += 1
            if len(diagnostics["tickerErrors"]) < 30:
                diagnostics["tickerErrors"].append({"ticker": ticker, "error": str(exc)[:200]})
            logger.exception("Error scanning ticker %s", ticker)

    return {
        "status": "success",
        "scanTime": _now_ist(),
        "totalMatches": len(matches),
        "data": matches,
        "diagnostics": diagnostics,
        "message": ("Scan completed using configured Swing Watchlist and daily market data." if matches else
                   "No configured watchlist stocks matched all five conditions; see diagnostics for data/skipped counts.")
    }
