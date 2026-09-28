import datetime
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

def get_nifty500_tickers():
    """NSE ನಿಂದ Nifty 500 ಶೇರುಗಳ ಲೈವ್ ಪಟ್ಟಿಯನ್ನು ಫೆಚ್ ಮಾಡುತ್ತದೆ"""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(
            url,
            storage_options={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        tickers = [f"{symbol}.NS" for symbol in df["Symbol"].dropna()]
        print(f"✅ Nifty 500 ಪಟ್ಟಿಯಿಂದ ಒಟ್ಟು {len(tickers)} ಶೇರುಗಳನ್ನು ಲೋಡ್ ಮಾಡಲಾಗಿದೆ.\n")
        return tickers
    except Exception as e:
        print(f"⚠️ Nifty 500 ಲಿಸ್ಟ್ ಸಿಗಲಿಲ್ಲ ({e}). ಡೀಫಾಲ್ಟ್ ಬ್ಯಾಕಪ್ ಲಿಸ್ಟ್ ಬಳಸಲಾಗುತ್ತಿದೆ...")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS"]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_inflow_scanner_v2():
    capital_per_trade = 30000.0  # ₹30,000 Capital
    sl_pct = 0.03                # -3.0% Stop Loss
    target_pct = 0.045           # +4.5% Target

    tickers = get_nifty500_tickers()

    print("==========================================================================")
    print("   INFLOW SCANNER: NIFTY 500 LIVE MARKET SCANNER")
    print(f"   Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================================================================")
    print("ನಿರೀಕ್ಷಿಸಿ... 500 ಶೇರುಗಳ ಲೈವ್ ಡೇಟಾವನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಲಾಗುತ್ತಿದೆ...\n")

    try:
        data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=True)
    except Exception as e:
        print(f"❌ ಡೇಟಾ ಡೌನ್‌ಲೋಡ್ ಫೇಲ್ ಆಗಿದೆ: {e}")
        return

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
                    actual_inv = qty * buy_price
                    matches.append({
                        "Stock": ticker.replace(".NS", ""),
                        "Buy Price (₹)": round(buy_price, 2),
                        "StopLoss (₹)": round(sl_price, 2),
                        "Target (₹)": round(target_price, 2),
                        "Qty": qty,
                        "RSI": round(float(curr["RSI_14"]), 1),
                        "Investment (₹)": round(actual_inv, 2),
                        "Max Risk (₹)": round(actual_inv * sl_pct, 2),
                        "Target Profit (₹)": round(actual_inv * target_pct, 2),
                    })
        except Exception:
            continue

    print("\n==========================================================================")
    if matches:
        res_df = pd.DataFrame(matches)
        print(f"🎯 ಒಟ್ಟು {len(res_df)} ಉತ್ತಮ ಶೇರುಗಳು ನಿಮ್ಮ ನಿಯಮಗಳಿಗೆ ಹೊಂದಿಕೆಯಾಗಿವೆ:\n")
        display(res_df)  # Google Colab ನಲ್ಲಿ Table ಸುಂದರವಾಗಿ ಕಾಣಲು
    else:
        print("❌ ಇವತ್ತು Nifty 500 ನ ಯಾವುದೇ ಶೇರು ನಿಮ್ಮ 5 ಎಂಟ್ರಿ ನಿಯಮಗಳನ್ನು ಪೂರೈಸಿಲ್ಲ.")
    print("==========================================================================")

# Execute Scanner
run_inflow_scanner_v2()
