import pandas as pd
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings('ignore')

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = -loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_inflow_scanner(capital_per_trade=30000.0):
    # Demo/Core Tickers (Fast scanning gagi, Nifty 500 list kooda add madabahudu)
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "TATAMOTORS.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LT.NS"
    ]
    
    sl_pct = 0.03       # -3.0% Stop Loss
    target_pct = 0.045  # +4.5% Target

    try:
        data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False)
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}

    matches = []

    for ticker in tickers:
        try:
            if ticker not in data or data[ticker].dropna().empty:
                continue

            df = data[ticker].dropna()
            if len(df) < 200:
                continue

            df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
            df["RSI_14"] = calculate_rsi(df["Close"], period=14)
            df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # 5 Core Rules
            c1 = (curr["Close"] > curr["EMA_200"]) or (curr["EMA_50"] > curr["EMA_200"])
            near_ema20 = abs(curr["Close"] - curr["EMA_20"]) / curr["EMA_20"] <= 0.02
            near_ema50 = abs(curr["Close"] - curr["EMA_50"]) / curr["EMA_50"] <= 0.02
            c2 = near_ema20 or near_ema50
            c3 = (curr["Close"] > curr["Open"]) and (curr["Close"] > prev["Close"])
            c4 = 38 <= curr["RSI_14"] <= 68
            c5 = curr["Volume"] >= curr["Vol_SMA20"]

            if c1 and c2 and c3 and c4 and c5:
                buy_price = float(curr["Close"])
                sl_price = buy_price * (1 - sl_pct)
                target_price = buy_price * (1 + target_pct)
                qty = int(capital_per_trade // buy_price)

                if qty >= 1:
                    matches.append({
                        "stock": ticker.replace(".NS", ""),
                        "buyPrice": round(buy_price, 2),
                        "stopLoss": round(sl_price, 2),
                        "target": round(target_price, 2),
                        "qty": qty,
                        "rsi": round(float(curr["RSI_14"]), 1),
                        "investment": round(qty * buy_price, 2)
                    })
        except Exception:
            continue

    return {
        "status": "success",
        "scanTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "totalMatches": len(matches),
        "data": matches
    }