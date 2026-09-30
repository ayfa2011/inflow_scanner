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
TARGET_PCT = 0.05


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


def calculate_atr(df, period=14):
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_adx(df, period=14):
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_supertrend(df, period=10, multiplier=3.0):
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    atr = calculate_atr(df, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    st = pd.Series(index=df.index, dtype="float64")
    bullish = pd.Series(index=df.index, dtype="bool")
    for i in range(len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(st.iloc[i - 1]):
            bullish.iloc[i] = True
            st.iloc[i] = final_lower.iloc[i]
            continue
        prev_close = close.iloc[i - 1]
        final_upper.iloc[i] = (upper.iloc[i] if upper.iloc[i] < final_upper.iloc[i - 1] or prev_close > final_upper.iloc[i - 1] else final_upper.iloc[i - 1])
        final_lower.iloc[i] = (lower.iloc[i] if lower.iloc[i] > final_lower.iloc[i - 1] or prev_close < final_lower.iloc[i - 1] else final_lower.iloc[i - 1])
        prev_bull = bool(bullish.iloc[i - 1])
        if prev_bull:
            bullish.iloc[i] = not (close.iloc[i] < final_lower.iloc[i])
        else:
            bullish.iloc[i] = close.iloc[i] > final_upper.iloc[i]
        st.iloc[i] = final_lower.iloc[i] if bullish.iloc[i] else final_upper.iloc[i]
    return st, bullish


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
    """Scan daily candles against the five original conditions plus Supertrend and ADX."""
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
            "Volume >= 20-day average volume",
            "Supertrend(10,3) bullish (Close above Supertrend line)",
            "ADX(14) > 25"
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
            required = {"Open", "High", "Low", "Close", "Volume"}
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
            df["ADX_14"] = calculate_adx(df, 14)
            df["Supertrend_10_3"], df["Supertrend_Bullish"] = calculate_supertrend(df, 10, 3.0)

            curr, prev = df.iloc[-1], df.iloc[-2]
            keys = ["Close", "Open", "EMA_20", "EMA_50", "EMA_200", "RSI_14", "Volume", "Vol_SMA20", "ADX_14", "Supertrend_10_3"]
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
            adx = float(curr["ADX_14"])
            supertrend_line = float(curr["Supertrend_10_3"])
            c6 = bool(curr["Supertrend_Bullish"]) and price > supertrend_line
            c7 = adx > 25

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                qty = int(capital // price)
                if qty >= 1:
                    invested = round(qty * price, 2)
                    info = ticker_info.get(ticker, {})
                    # Transparent rule-fit score (0-100), not a probability of profit.
                    ema_distance = min(abs(price - ema20) / ema20, abs(price - ema50) / ema50)
                    proximity_score = max(0.0, 20.0 * (1.0 - ema_distance / 0.02))
                    volume_ratio = vol / vol_avg if vol_avg > 0 else 0.0
                    volume_score = min(15.0, max(0.0, volume_ratio * 10.0))
                    trend_score = (10.0 if ema50 > ema200 else 0.0) + (5.0 if price > ema200 else 0.0)
                    rsi_score = max(0.0, 10.0 * (1.0 - abs(rsi - 53.0) / 30.0))
                    score = round(min(100.0, 50.0 + proximity_score + volume_score + trend_score + rsi_score), 1)
                    reasons = [
                        "Trend filter passed" if c1 else "Trend filter failed",
                        "Price within 2% of EMA20/EMA50",
                        "Bullish daily candle and above previous close",
                        "RSI within 38–68",
                        "Volume at/above 20-day average",
                        "Supertrend(10,3) bullish; close above line",
                        "ADX(14) > 25",
                    ]
                    matches.append({
                        "stock": ticker.removesuffix(".NS"),
                        "companyName": info.get("name", ticker.removesuffix(".NS")),
                        "sector": info.get("sector", ""),
                        "buyPrice": round(price, 2),
                        "qty": qty,
                        "investment": invested,
                        "stopLoss": None,
                        "target": round(price * (1 + TARGET_PCT), 2),
                        "targetPct": 5,
                        "rsi": round(rsi, 1),
                        "supertrend": round(supertrend_line, 2),
                        "adx": round(adx, 2),
                        "conditionsPassed": 7,
                        "maxRisk": None,
                        "targetProfit": round(invested * TARGET_PCT, 2),
                        "setupScore": score,
                        "scoreMeaning": "Rule-fit score only; not a win probability or return forecast.",
                        "setupReasons": reasons,
                        "volumeRatio": round(volume_ratio, 2),
                        "ema20DistancePct": round(abs(price - ema20) / ema20 * 100, 2),
                        "ema50DistancePct": round(abs(price - ema50) / ema50 * 100, 2),
                    })
        except Exception as exc:
            diagnostics["tickersSkipped"] += 1
            if len(diagnostics["tickerErrors"]) < 30:
                diagnostics["tickerErrors"].append({"ticker": ticker, "error": str(exc)[:200]})
            logger.exception("Error scanning ticker %s", ticker)

    matches.sort(key=lambda item: item.get("setupScore", 0), reverse=True)

    return {
        "status": "success",
        "scanTime": _now_ist(),
        "totalMatches": len(matches),
        "data": matches,
        "diagnostics": diagnostics,
        "message": ("Scan completed using configured Swing Watchlist and daily market data." if matches else
                   "No configured watchlist stocks matched all seven conditions; see diagnostics for data/skipped counts."),
        "rankingNote": "Sorted by transparent rule-fit score; score is not a probability of profit.",
        "exitPlan": "Existing exit behavior unchanged; no stop-loss or trailing-stop management is included in this scanner."
    }
