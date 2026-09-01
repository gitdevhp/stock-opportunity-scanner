import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_universe(cfg):
    tickers = set(cfg.get("watchlist_additions", []))
    excludes = set(cfg.get("exclude", []))

    # S&P 500
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers.update(sp)
    except Exception as e:
        print("S&P universe fetch failed:", e)

    # Nasdaq-100
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ["Ticker", "Ticker symbol", "Symbol"]:
                if col in t.columns:
                    tickers.update(t[col].astype(str).tolist())
    except Exception as e:
        print("Nasdaq universe fetch failed:", e)

    return sorted(t for t in tickers if t and t not in excludes and not t.startswith("^"))


def download_prices(tickers, period_days=420):
    # yfinance supports batch downloads. We use daily bars for a medium-term system.
    period = "2y" if period_days > 500 else "1y"
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )
    return df


def series_for(df, field, ticker):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if field in df.columns.get_level_values(0):
                s = df[field]
                if ticker in s.columns:
                    return s[ticker].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def linreg_slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def pivots(s, order=3):
    vals = s.values
    lows, highs = [], []
    for i in range(order, len(vals)-order):
        w = vals[i-order:i+order+1]
        if vals[i] == np.min(w):
            lows.append((i, vals[i]))
        if vals[i] == np.max(w):
            highs.append((i, vals[i]))
    return lows, highs


def cluster_levels(points, tolerance=0.012):
    if not points:
        return []
    vals = sorted(float(v) for _, v in points)
    clusters = []
    for v in vals:
        if not clusters or abs(v - np.mean(clusters[-1])) / max(np.mean(clusters[-1]), 1e-9) > tolerance:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [(float(np.mean(c)), len(c)) for c in clusters]


def fib_levels(high, low):
    d = high - low
    return {
        "23.6": high - 0.236*d,
        "38.2": high - 0.382*d,
        "50.0": high - 0.500*d,
        "61.8": high - 0.618*d,
        "78.6": high - 0.786*d,
    }


def analyze_ticker(ticker, df, cfg):
    if len(df) < 220:
        return None

    close = df["Close"].astype(float).dropna()
    high = df["High"].astype(float).reindex(close.index)
    low = df["Low"].astype(float).reindex(close.index)
    volume = df["Volume"].astype(float).reindex(close.index).fillna(0)

    if len(close) < 220:
        return None

    price = float(close.iloc[-1])
    avg_vol = float(volume.tail(50).mean())
    if price < cfg["min_price"] or avg_vol < cfg["min_average_volume"]:
        return None

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    m50, m200 = float(ma50.iloc[-1]), float(ma200.iloc[-1])
    prev50, prev200 = float(ma50.iloc[-6]), float(ma200.iloc[-6])

    r = rsi(close)
    rsi_now = float(r.iloc[-1])

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    bb_now = (price - float(lower.iloc[-1])) / max(float(upper.iloc[-1]-lower.iloc[-1]), 1e-9)
    bb_width = float((upper.iloc[-1]-lower.iloc[-1]) / max(mid.iloc[-1], 1e-9))
    bb_width_20 = float(((upper-mid)/mid).tail(20).mean()*2)

    mline, msig, mhist = macd(close)
    macd_now = float(mline.iloc[-1] - msig.iloc[-1])
    macd_prev = float(mline.iloc[-2] - msig.iloc[-2])

    atrs = atr(pd.DataFrame({"High":high,"Low":low,"Close":close}))
    atr_now = float(atrs.iloc[-1])
    atr_pct = atr_now / price * 100

    # Major horizontal levels from local pivots in the last 180 sessions.
    recent = close.tail(180)
    ph = high.loc[recent.index]
    pl = low.loc[recent.index]
    lows, highs = pivots(pl, 3), pivots(ph, 3)
    low_clusters = cluster_levels(lows)
    high_clusters = cluster_levels(highs)

    supports = [(v,c) for v,c in low_clusters if v < price*1.01]
    resistances = [(v,c) for v,c in high_clusters if v > price*0.99]
    supports = sorted(supports, key=lambda x: abs(price-x[0]))[:5]
    resistances = sorted(resistances, key=lambda x: abs(price-x[0]))[:5]

    support = max([v for v,c in supports if v <= price*1.015], default=float(low.tail(20).min()))
    resistance = min([v for v,c in resistances if v >= price*0.985], default=float(high.tail(20).max()))

    # 60-day channel: regression centerline + residual band.
    ch = close.tail(60)
    slope = linreg_slope(ch.values)
    x = np.arange(len(ch))
    fit = np.polyval(np.polyfit(x, ch.values, 1), x)
    resid = ch.values - fit
    channel_sigma = float(np.std(resid))
    center_now = float(fit[-1])
    ch_upper = center_now + 2*channel_sigma
    ch_lower = center_now - 2*channel_sigma
    slope_pct = slope / price * 100

    # Convergence: compare channel width now vs 20 sessions ago.
    widths = []
    for n in [60, 40, 20]:
        ss = close.tail(n)
        xx = np.arange(len(ss))
        ff = np.polyval(np.polyfit(xx, ss.values, 1), xx)
        widths.append(float(np.std(ss.values-ff)))
    converging = widths[-1] < widths[0]*0.85

    # Breakout and volume confirmation.
    prior20_high = float(high.iloc[-21:-1].max())
    prior20_low = float(low.iloc[-21:-1].min())
    vol_ratio = float(volume.iloc[-1] / max(volume.tail(20).mean(), 1))
    breakout_up = price > prior20_high * 1.002
    breakdown = price < prior20_low * 0.998
    breakout_volume = breakout_up and vol_ratio >= 1.25

    # Golden/death cross.
    golden = m50 > m200 and prev50 <= prev200
    death = m50 < m200 and prev50 >= prev200
    bullish_ma = m50 > m200 and price > m50
    bearish_ma = m50 < m200 and price < m50

    # Fib using 120-session swing high/low.
    swing_high = float(high.tail(120).max())
    swing_low = float(low.tail(120).min())
    fib = fib_levels(swing_high, swing_low)

    def near(level, pct=0.025):
        return abs(price-level)/max(abs(level),1e-9) <= pct

    # Scoring.
    score = 0.0
    reasons = []
    major_count = 0

    # MA relationship — 20
    if golden:
        score += 20; reasons.append("fresh golden cross"); major_count += 1
    elif bullish_ma:
        score += 15; reasons.append("bullish 50/200 structure")
    elif death:
        score += 0
        reasons.append("fresh death cross")
    elif bearish_ma:
        score += 2
        reasons.append("bearish 50/200 structure")
    else:
        score += 8

    # Horizontal S/R — 20
    if support > 0 and near(support, 0.018):
        score += 20; reasons.append("price testing repeated support"); major_count += 1
    elif resistance > 0 and near(resistance, 0.018):
        score += 16; reasons.append("price testing repeated resistance")
    elif support > 0 and price > support:
        score += 10
    else:
        score += 4

    # Channel — 20
    if converging:
        score += 15; reasons.append("converging channel"); major_count += 1
    else:
        score += 7
    if slope > 0:
        score += 5; reasons.append("positive channel slope")
    elif slope < 0:
        score += 1

    # Breakout/retest + volume — 20
    if breakout_volume:
        score += 20; reasons.append("volume-confirmed breakout"); major_count += 1
    elif breakout_up:
        score += 13; reasons.append("breakout without strong volume")
    elif price > prior20_high*0.985:
        score += 8; reasons.append("approaching breakout resistance")
    elif price > support:
        score += 5

    # Secondary 20
    # Bollinger 5
    if bb_now <= 0.25:
        score += 5; reasons.append("near lower Bollinger band")
    elif bb_now <= 0.45:
        score += 4
    elif 0.45 < bb_now < 0.75:
        score += 3
    elif bb_now >= 0.95:
        score += 0; reasons.append("near upper Bollinger band")
    else:
        score += 2

    # RSI 5
    if 45 <= rsi_now <= 62:
        score += 5
    elif 38 <= rsi_now < 45 or 62 < rsi_now <= 68:
        score += 3
    elif rsi_now < 30:
        score += 2
    else:
        score += 1

    # Fibonacci 5
    fib_dist = min(abs(price-v)/price for v in fib.values())
    if fib_dist < 0.012:
        score += 5; reasons.append("near key Fibonacci retracement")
    elif fib_dist < 0.025:
        score += 3
    else:
        score += 1

    # MACD/volatility 5
    if macd_now > 0 and macd_now >= macd_prev:
        score += 5
    elif macd_now > 0:
        score += 3
    else:
        score += 1

    # Penalties
    if atr_pct > cfg["max_atr_pct"]:
        score -= 5
        reasons.append("high ATR penalty")
    if rsi_now > 72:
        score -= 6
        reasons.append("overbought penalty")
    if rsi_now < 25:
        score -= 3
    if breakdown:
        score -= 12
        reasons.append("major breakdown penalty")

    score = max(0, min(100, score))

    # Entry logic: prefer pullback to support if it is reasonably close;
    # otherwise use breakout confirmation.
    if support > 0 and support < price and (price-support)/price <= 0.08:
        entry = support * 1.003
        entry_type = "support-retest limit"
    elif resistance > price:
        entry = resistance * 1.002
        entry_type = "breakout confirmation"
    else:
        entry = price

    # Make entry less aggressive if current price is extended from support.
    if entry > price*1.03 and resistance > price:
        entry = resistance*1.002
        entry_type = "breakout confirmation"

    # Stop based on support/ATR.
    stop_candidates = [support*0.985 if support>0 else np.nan, price-2.0*atr_now]
    stop = max([x for x in stop_candidates if np.isfinite(x) and x < price], default=price*0.93)

    # Target: nearest resistance and 2R, capped by a reasonable technical level.
    r1 = resistance if resistance > entry*1.01 else prior20_high
    target_2r = entry + 2*(entry-stop)
    target = max(r1, target_2r)
    if target <= entry:
        target = entry + 3*atr_now

    capital = cfg["capital"]
    shares = capital / entry if entry > 0 else 0
    risk_dollars = max(entry-stop, 0)*shares
    risk_pct = max(entry-stop, 0)/entry*100 if entry else 0

    # Setup state
    if score >= 85 and major_count >= 2:
        state = "HIGH CONVICTION"
    elif score >= 75 and major_count >= 2:
        state = "WATCH / PREPARE LIMIT"
    elif score >= 65:
        state = "DEVELOPING"
    else:
        state = "IGNORE"

    # If entry is a breakout level, don't call it a current buy.
    if entry_type == "breakout confirmation" and not breakout_up:
        state = "WAIT FOR CONFIRMATION" if score >= 65 else state

    return {
        "ticker": ticker,
        "price": price,
        "score": round(score,1),
        "state": state,
        "entry": round(entry,2),
        "entry_type": entry_type,
        "breakout": round(resistance*1.002,2) if resistance else None,
        "stop": round(stop,2),
        "target": round(target,2),
        "shares": round(shares,4),
        "risk_dollars": round(risk_dollars,2),
        "risk_pct": round(risk_pct,2),
        "ma50": round(m50,2),
        "ma200": round(m200,2),
        "golden": bool(golden or bullish_ma),
        "death": bool(death or bearish_ma),
        "support": round(support,2),
        "resistance": round(resistance,2),
        "rsi": round(rsi_now,1),
        "bb_position": round(bb_now,2),
        "bb_width": round(bb_width,3),
        "atr_pct": round(atr_pct,2),
        "volume_ratio": round(vol_ratio,2),
        "converging": bool(converging),
        "channel_slope_pct": round(slope_pct,3),
        "fib_38": round(fib["38.2"],2),
        "fib_50": round(fib["50.0"],2),
        "fib_61": round(fib["61.8"],2),
        "macd_hist": round(macd_now,4),
        "major_count": major_count,
        "reasons": reasons[:8],
    }


def format_report(results, mode, cfg):
    now = datetime.now(ZoneInfo(cfg["timezone"]))
    title = f"📊 {mode.upper()} STOCK OPPORTUNITY SCAN — {now:%Y-%m-%d %I:%M %p %Z}"

    lines = [title, "", f"Capital model: ${cfg['capital']:.2f}",
             "Framework: MA 20% | S/R 20% | Channel 20% | Breakout+volume 20% | Secondary 20%",
             ""]

    if not results:
        lines += ["⚪ NO QUALIFYING SETUPS", "", "No stock currently meets the minimum technical threshold. Do not force a trade."]
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        emoji = "🔥" if r["score"] >= 85 else "🟢" if r["score"] >= 75 else "🟡"
        lines += [
            f"{emoji} #{i} {r['ticker']} — {r['score']}/100 — {r['state']}",
            f"Price ${r['price']:.2f} | Entry ${r['entry']:.2f} ({r['entry_type']})",
            f"Confirmation ${r['breakout']:.2f} | Stop ${r['stop']:.2f} | Target ${r['target']:.2f}",
            f"Position at ${cfg['capital']:.2f}: {r['shares']:.4f} shares | modeled risk ${r['risk_dollars']:.2f} ({r['risk_pct']:.1f}%)",
            "",
            "OBSERVE",
            f"50MA ${r['ma50']:.2f} | 200MA ${r['ma200']:.2f} | "
            f"{'Golden/bullish' if r['golden'] else 'Death/bearish' if r['death'] else 'Mixed'}",
            f"Support ${r['support']:.2f} | Resistance ${r['resistance']:.2f} | "
            f"Channel {'converging' if r['converging'] else 'not strongly converging'}",
            f"RSI {r['rsi']:.1f} | BB position {r['bb_position']:.2f} | "
            f"ATR {r['atr_pct']:.1f}% | Volume {r['volume_ratio']:.2f}x",
            f"Fib 38.2 ${r['fib_38']:.2f} | 50 ${r['fib_50']:.2f} | 61.8 ${r['fib_61']:.2f}",
            "",
            "REASON",
            "; ".join(r["reasons"]),
            "",
            "RESPONSE",
            f"Preferred action: {r['state']}. Exact modeled entry: ${r['entry']:.2f}. "
            f"Do not chase above the confirmation level without volume.",
            "—" * 34,
            ""
        ]

    return "\n".join(lines)


def send_discord(content, webhook):
    # Discord messages have a content limit; split safely.
    chunks = []
    while len(content) > 1900:
        cut = content.rfind("\n", 0, 1900)
        if cut < 500:
            cut = 1900
        chunks.append(content[:cut])
        content = content[cut:].lstrip()
    chunks.append(content)

    for chunk in chunks:
        r = requests.post(webhook, json={"content": chunk}, timeout=20)
        r.raise_for_status()


def should_run(mode, cfg):
    now = datetime.now(ZoneInfo(cfg["timezone"]))
    now_minutes = now.hour*60 + now.minute
    if mode == "intraday":
        # Hourly scans from 09:00 through 15:00 local Central Time.
        # Allow a 20-minute window so a delayed GitHub runner still sends.
        return any(abs(now_minutes - h*60) <= 20 for h in range(9, 16))
    target = (cfg["morning_local_hour"], cfg["morning_local_minute"]) if mode == "morning" else (cfg["evening_local_hour"], cfg["evening_local_minute"])
    # Allow a 25-minute window because GitHub scheduled jobs can start late.
    target_minutes = target[0]*60 + target[1]
    return abs(now_minutes-target_minutes) <= 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["morning","intraday","evening"], default="evening")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--force-time", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if not args.force_time and not should_run(args.mode, cfg) and os.getenv("GITHUB_ACTIONS") == "true":
        print("Outside configured local-time window; exiting without sending.")
        return

    tickers = get_universe(cfg)
    print(f"Scanning {len(tickers)} tickers...")

    data = download_prices(tickers, cfg["history_days"])
    results = []

    # DataFrame layout from yfinance is MultiIndex: field x ticker.
    for ticker in tickers:
        try:
            sub = pd.DataFrame({
                "Open": series_for(data, "Open", ticker),
                "High": series_for(data, "High", ticker),
                "Low": series_for(data, "Low", ticker),
                "Close": series_for(data, "Close", ticker),
                "Volume": series_for(data, "Volume", ticker),
            }).dropna(subset=["Close"])
            if len(sub) >= 220:
                r = analyze_ticker(ticker, sub, cfg)
                if r and r["score"] >= cfg["min_score_to_report"]:
                    results.append(r)
        except Exception as e:
            print(f"{ticker}: {e}")

    results.sort(key=lambda x: (x["score"], x["major_count"], x["volume_ratio"]), reverse=True)
    results = results[:cfg["max_candidates"]]

    report = format_report(results, args.mode, cfg)
    print(report)

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(ROOT, "reports", f"{stamp}_{args.mode}.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    if args.send:
        webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook:
            print("DISCORD_WEBHOOK_URL missing; report printed but not sent.")
            return
        send_discord(report, webhook)


if __name__ == "__main__":
    main()
