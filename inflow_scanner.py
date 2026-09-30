"""Inflow Swing Pick scanner: original 5 checks + Supertrend(10,3) + ADX(14)>=25.
Daily OHLCV from yfinance; not guaranteed real-time. Stateless scanner only returns an
initial ATR stop reference. A live trailing stop must be ratcheted per open position.
"""
import datetime as dt
import json, logging, math
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

logger = logging.getLogger("inflow_scanner")
WATCHLIST_PATH = Path(__file__).resolve().with_name("swing_watchlist.json")
TARGET_PCT = 0.05
ST_PERIOD = 10
ST_MULTIPLIER = 3.0
ADX_PERIOD = 14
ADX_MIN = 25.0
ATR_TRAIL_MULTIPLIER = 3.0


def _now_ist():
    return dt.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")


def load_swing_watchlist(path=None):
    with (Path(path) if path else WATCHLIST_PATH).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    configured, unconfigured, seen = [], [], set()
    for item in payload.get("stocks", []):
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        name = str(item.get("name", "")).strip()
        symbol = str(item.get("nse_symbol") or "").strip().upper()
        if not name or not symbol:
            unconfigured.append({"name": name or "(unnamed)", "reason": "Name or nse_symbol missing"})
            continue
        ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
        if ticker in seen: continue
        seen.add(ticker)
        configured.append({"name": name, "sector": item.get("sector", ""), "symbol": symbol.removesuffix(".NS"), "ticker": ticker})
    return configured, unconfigured


def _ticker_frame(data, ticker):
    if data is None or data.empty: return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex): return data.copy()
    if ticker in data.columns.get_level_values(0): frame = data[ticker].copy()
    elif ticker in data.columns.get_level_values(1): frame = data.xs(ticker, axis=1, level=1).copy()
    else: return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex): frame.columns = frame.columns.get_level_values(0)
    return frame


def calculate_atr(df, period=10):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, float("nan"))
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=bool)
    close = df["Close"]
    for i in range(len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(final_upper.iloc[i-1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            direction.iloc[i] = True
            st.iloc[i] = final_lower.iloc[i]
            continue
        prev_u, prev_l = final_upper.iloc[i-1], final_lower.iloc[i-1]
        final_upper.iloc[i] = basic_upper.iloc[i] if (basic_upper.iloc[i] < prev_u or close.iloc[i-1] > prev_u) else prev_u
        final_lower.iloc[i] = basic_lower.iloc[i] if (basic_lower.iloc[i] > prev_l or close.iloc[i-1] < prev_l) else prev_l
        prev_dir = bool(direction.iloc[i-1]) if not pd.isna(direction.iloc[i-1]) else True
        if prev_dir:
            curr_dir = not (close.iloc[i] < final_lower.iloc[i])
        else:
            curr_dir = close.iloc[i] > final_upper.iloc[i]
        direction.iloc[i] = curr_dir
        st.iloc[i] = final_lower.iloc[i] if curr_dir else final_upper.iloc[i]
    return st, direction


def _rsi(series, period=14):
    d = series.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    al=loss.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    rs=ag/al.replace(0,float("nan")); out=100-100/(1+rs)
    return out.mask((al==0)&(ag>0),100).mask((al==0)&(ag==0),50)


def run_inflow_scanner(capital_per_trade=30000.0):
    try:
        capital=float(capital_per_trade)
        if not math.isfinite(capital) or capital<=0: raise ValueError
    except (TypeError, ValueError):
        return {"status":"error","message":"Capital must be a positive number.","data":[],"watchlist":[]}
    try:
        watchlist, unconfigured=load_swing_watchlist()
    except Exception as exc:
        logger.exception("Could not load watchlist")
        return {"status":"error","message":"Could not load swing_watchlist.json: "+str(exc)[:200],"scanTime":_now_ist(),"totalMatches":0,"data":[],"watchlist":[]}
    tickers=[x["ticker"] for x in watchlist]; info={x["ticker"]:x for x in watchlist}
    diag={"tickerSource":"swing_watchlist.json","watchlistEnabled":len(watchlist)+len(unconfigured),"tickersRequested":len(tickers),"unconfiguredStocks":unconfigured,"tickersWithData":0,"tickersInsufficientHistory":0,"tickersSkipped":0,"tickerErrors":[],"dataInterval":"1d","strategy":"Original five conditions + Supertrend(10,3) bullish + ADX(14)>=25"}
    labels=["Original trend: Close > EMA200 OR EMA50 > EMA200","Price within 2% of EMA20 or EMA50","Bullish candle and above previous close","RSI(14) between 38 and 68","Volume >= 20-day average","Supertrend(10,3) bullish","ADX(14) >= 25"]
    def pending_row(item, reason):
        return {"stock":item["symbol"],"companyName":item.get("name",item["symbol"]),"sector":item.get("sector",""),"matched":False,"status":"Pending — "+reason,"dataStatus":reason,"conditions":[{"label":lab,"pass":False} for lab in labels],"conditionPoints":0,"conditionTotal":7,"buyPrice":None,"qty":None,"investment":None,"target":None,"targetPct":5,"targetProfit":None,"rsi":None,"supertrend":None,"supertrendBullish":None,"adx":None,"atr10":None,"initialAtrStop":None,"setupScore":0,"scoreMeaning":"Rule checks passed (0-7), not a win probability or return forecast."}
    all_rows=[pending_row(item,"Waiting for market data") for item in watchlist]
    if not tickers:
        return {"status":"error","message":"No enabled stocks have configured nse_symbol values.","scanTime":_now_ist(),"totalMatches":0,"data":all_rows,"watchlist":all_rows,"diagnostics":diag}
    try:
        raw=yf.download(tickers=tickers,period="1y",interval="1d",group_by="ticker",auto_adjust=True,progress=False,threads=True,timeout=25)
    except Exception as exc:
        logger.exception("Market data download failed")
        msg="Market data download failed: "+str(exc)[:200]
        for row in all_rows: row["status"]="Pending — market data unavailable"; row["dataStatus"]=msg
        return {"status":"error","message":msg,"scanTime":_now_ist(),"totalMatches":0,"data":all_rows,"watchlist":all_rows,"diagnostics":diag}
    matches=[]; row_by_ticker={item["ticker"]:row for item,row in zip(watchlist,all_rows)}
    if raw is None or raw.empty:
        for row in all_rows: row["status"]="Pending — provider returned no data"; row["dataStatus"]="Market data provider returned no data"
        return {"status":"error","message":"Market data provider returned no data.","scanTime":_now_ist(),"totalMatches":0,"data":all_rows,"watchlist":all_rows,"diagnostics":diag}
    for ticker in tickers:
        row=row_by_ticker[ticker]
        try:
            df=_ticker_frame(raw,ticker)
            if df.empty or not all(col in df.columns for col in ["Open","High","Low","Close","Volume"]):
                diag["tickersSkipped"]+=1; row["status"]="Pending — no market data"; row["dataStatus"]="No OHLCV data returned for symbol"; continue
            df=df.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
            if len(df)<220:
                diag["tickersInsufficientHistory"]+=1; row["status"]="Pending — insufficient history"; row["dataStatus"]="Need at least 220 daily OHLCV rows; received "+str(len(df)); continue
            diag["tickersWithData"]+=1
            c=df["Close"].astype(float); vol=df["Volume"].astype(float)
            ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); ema200=c.ewm(span=200,adjust=False).mean()
            rsi=_rsi(c,14); volavg=vol.rolling(20,min_periods=20).mean(); atr=calculate_atr(df,ST_PERIOD); adx=calculate_adx(df,ADX_PERIOD)
            st_line, st_bull=calculate_supertrend(df,ST_PERIOD,ST_MULTIPLIER)
            cur=df.iloc[-1]; prev=df.iloc[-2]; price=float(cur["Close"]); atr_now=float(atr.iloc[-1]); adx_now=float(adx.iloc[-1]); rsi_now=float(rsi.iloc[-1])
            if not all(math.isfinite(x) for x in [price,atr_now,adx_now,rsi_now]) or price<=0 or atr_now<=0 or pd.isna(volavg.iloc[-1]):
                diag["tickersSkipped"]+=1; row["status"]="Pending — indicator data unavailable"; row["dataStatus"]="Required indicator values are unavailable"; continue
            cond=[bool(price>ema200.iloc[-1] or ema50.iloc[-1]>ema200.iloc[-1]),bool(abs(price-ema20.iloc[-1])/ema20.iloc[-1]<=.02 or abs(price-ema50.iloc[-1])/ema50.iloc[-1]<=.02),bool(price>float(cur["Open"]) and price>float(prev["Close"])),bool(38<=rsi_now<=68),bool(float(cur["Volume"])>=float(volavg.iloc[-1])),bool(st_bull.iloc[-1]),bool(adx_now>=ADX_MIN)]
            pts=sum(cond); matched=all(cond); qty=int(capital//price); invested=round(qty*price,2)
            row.update({"matched":matched,"status":"Setup complete" if matched else "Setup not complete","dataStatus":"Market data available","conditions":[{"label":lab,"pass":ok} for lab,ok in zip(labels,cond)],"conditionPoints":pts,"conditionTotal":7,"buyPrice":round(price,2),"qty":qty if qty>0 else None,"investment":invested if qty>0 else None,"target":round(price*(1+TARGET_PCT),2),"targetPct":5,"targetProfit":round(invested*TARGET_PCT,2) if qty>0 else None,"rsi":round(rsi_now,1),"supertrend":round(float(st_line.iloc[-1]),2),"supertrendBullish":bool(st_bull.iloc[-1]),"adx":round(adx_now,2),"atr10":round(atr_now,2),"initialAtrStop":round(max(.01,price-ATR_TRAIL_MULTIPLIER*atr_now),2),"atrTrailMultiplier":ATR_TRAIL_MULTIPLIER,"setupScore":pts,"scoreMeaning":"Rule checks passed (0-7), not a win probability or return forecast.","volumeRatio":round(float(cur["Volume"])/float(volavg.iloc[-1]),2) if float(volavg.iloc[-1])>0 else None})
            if matched and qty>0: matches.append(row)
        except Exception as exc:
            diag["tickersSkipped"]+=1
            row["status"]="Pending — scan error"; row["dataStatus"]=str(exc)[:180]
            if len(diag["tickerErrors"])<30: diag["tickerErrors"].append({"ticker":ticker,"error":str(exc)[:180]})
            logger.exception("Ticker scan error: %s",ticker)
    all_rows.sort(key=lambda x:(x["conditionPoints"],x.get("adx") or 0),reverse=True)
    matches.sort(key=lambda x:(x["conditionPoints"],x.get("adx") or 0),reverse=True)
    return {"status":"success","scanTime":_now_ist(),"totalMatches":len(matches),"data":all_rows,"watchlist":all_rows,"matches":matches,"diagnostics":diag,"message":"Every enabled configured stock is listed; missing data is marked Pending. Setups require all 7 checks.","rankingNote":"Sorted by checks passed then ADX; this is not a profit probability ranking.","strategy":{"supertrendPeriod":10,"supertrendMultiplier":3,"adxPeriod":14,"adxMinimum":25,"atrPeriod":10,"atrTrailMultiplier":3,"targetPct":5}}
