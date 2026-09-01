import argparse
import json
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))

CORE_WEIGHTS = {"ma": 15, "sr": 30, "channel": 20, "breakout": 15, "secondary": 20}


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_universe(cfg):
    tickers = set(cfg.get("watchlist_additions", []))
    excludes = set(cfg.get("exclude", []))
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers.update(sp)
    except Exception as e:
        print("S&P universe fetch failed:", e)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ["Ticker", "Ticker symbol", "Symbol"]:
                if col in t.columns:
                    tickers.update(t[col].astype(str).tolist())
    except Exception as e:
        print("Nasdaq universe fetch failed:", e)
    return sorted(t for t in tickers if t and t not in excludes and not t.startswith("^"))


def download_prices(tickers, period_days=420, intraday=False):
    if intraday:
        return yf.download(
            tickers=tickers, period="5d", interval="15m", auto_adjust=True,
            group_by="column", threads=True, progress=False, prepost=False,
        )
    period = "2y" if period_days > 500 else "1y"
    return yf.download(
        tickers=tickers, period=period, interval="1d", auto_adjust=True,
        group_by="column", threads=True, progress=False,
    )


def series_for(df, field, ticker):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if field in df.columns.get_level_values(0):
                s = df[field]
                if ticker in s.columns:
                    return s[ticker].dropna()
        elif field in df.columns:
            return df[field].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(s):
    line = ema(s, 12) - ema(s, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def linreg(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    return coef[0], np.polyval(coef, x)


def pivot_points(s, order=3):
    vals = np.asarray(s, dtype=float)
    lows, highs = [], []
    for i in range(order, len(vals) - order):
        w = vals[i - order:i + order + 1]
        if np.isfinite(vals[i]) and vals[i] == np.min(w):
            lows.append((i, float(vals[i])))
        if np.isfinite(vals[i]) and vals[i] == np.max(w):
            highs.append((i, float(vals[i])))
    return lows, highs


def cluster_levels(points, tolerance=0.012):
    vals = sorted(float(v) for _, v in points)
    if not vals:
        return []
    clusters = []
    for v in vals:
        if not clusters or abs(v - np.mean(clusters[-1])) / max(abs(np.mean(clusters[-1])), 1e-9) > tolerance:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [(float(np.mean(c)), len(c)) for c in clusters]


def fib_levels(high, low):
    d = max(high - low, 1e-9)
    return {"23.6": high - .236*d, "38.2": high - .382*d, "50.0": high - .500*d,
            "61.8": high - .618*d, "78.6": high - .786*d}


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))


def analyze_ticker(ticker, df, cfg, intraday_df=None):
    if len(df) < 220:
        return None
    close = df["Close"].astype(float).dropna()
    high = df["High"].astype(float).reindex(close.index)
    low = df["Low"].astype(float).reindex(close.index)
    volume = df["Volume"].astype(float).reindex(close.index).fillna(0)
    if len(close) < 220:
        return None

    daily_price = float(close.iloc[-1])
    price = daily_price
    intraday_last = None
    intraday_vol_ratio = None
    if intraday_df is not None and not intraday_df.empty:
        ic = series_for(intraday_df, "Close", ticker)
        iv = series_for(intraday_df, "Volume", ticker)
        if len(ic):
            intraday_last = float(ic.iloc[-1])
            if np.isfinite(intraday_last):
                price = intraday_last
        if len(iv) >= 20:
            intraday_vol_ratio = float(iv.iloc[-1] / max(iv.tail(20).mean(), 1))

    avg_vol = float(volume.tail(50).mean())
    if price < cfg["min_price"] or avg_vol < cfg["min_average_volume"]:
        return None

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    m50, m200 = float(ma50.iloc[-1]), float(ma200.iloc[-1])
    prev50, prev200 = float(ma50.iloc[-6]), float(ma200.iloc[-6])
    golden = m50 > m200 and prev50 <= prev200
    death = m50 < m200 and prev50 >= prev200
    bullish_ma = m50 > m200
    price_above_50 = price >= m50

    rsi_now = float(rsi(close).iloc[-1])
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bb_u, bb_l, bb_m = float(upper.iloc[-1]), float(lower.iloc[-1]), float(mid.iloc[-1])
    bb_pos = (price - bb_l) / max(bb_u - bb_l, 1e-9)
    bb_width = (bb_u - bb_l) / max(bb_m, 1e-9)

    mline, msig, mhist = macd(close)
    macd_hist_now = float(mhist.iloc[-1])
    macd_hist_prev = float(mhist.iloc[-2])

    atr_now = float(atr(pd.DataFrame({"High": high, "Low": low, "Close": close})).iloc[-1])
    atr_pct = atr_now / max(price, 1e-9) * 100

    # Repeated horizontal levels from confirmed daily pivots.
    recent = close.tail(180)
    ph, pl = high.loc[recent.index], low.loc[recent.index]
    low_points, _ = pivot_points(pl, 3)
    _, high_points = pivot_points(ph, 3)
    low_clusters = cluster_levels(low_points)
    high_clusters = cluster_levels(high_points)

    supports = sorted([(v, c) for v, c in low_clusters if v < price * 1.01],
                      key=lambda x: (abs(price - x[0]), -x[1]))[:8]
    resistances = sorted([(v, c) for v, c in high_clusters if v > price * 0.99],
                         key=lambda x: (abs(price - x[0]), -x[1]))[:8]
    support_candidates = [(v, c) for v, c in supports if v < price]
    resistance_candidates = [(v, c) for v, c in resistances if v > price]
    support = max(support_candidates, key=lambda x: (x[1], -abs(price-x[0])))[0] if support_candidates else float(low.tail(20).min())
    resistance = min(resistance_candidates, key=lambda x: (abs(price-x[0]), -x[1]))[0] if resistance_candidates else float(high.tail(20).max())
    support_touches = next((c for v, c in support_candidates if abs(v-support)/max(support,1e-9) < .012), 1)
    resistance_touches = next((c for v, c in resistance_candidates if abs(v-resistance)/max(resistance,1e-9) < .012), 1)

    # 60-day regression channel and a shorter 20-day channel for convergence.
    ch = close.tail(60)
    slope, fit = linreg(ch.values)
    sigma = float(np.std(ch.values - fit))
    center = float(fit[-1])
    ch_upper, ch_lower = center + 2*sigma, center - 2*sigma
    slope_pct = slope / max(price,1e-9) * 100
    short = close.tail(20)
    sslope, sfit = linreg(short.values)
    ssigma = float(np.std(short.values - sfit))
    short_width = 2*ssigma / max(float(short.iloc[-1]),1e-9)
    long_width = 2*sigma / max(price,1e-9)
    converging = short_width < long_width * .85
    near_lower_channel = price <= ch_lower * 1.03
    near_upper_channel = price >= ch_upper * .97

    prior20_high = float(high.iloc[-21:-1].max())
    prior20_low = float(low.iloc[-21:-1].min())
    daily_vol_ratio = float(volume.iloc[-1] / max(volume.tail(20).mean(), 1))
    breakout_up = price > prior20_high * 1.002
    breakdown = price < prior20_low * .998
    breakout_volume = breakout_up and daily_vol_ratio >= 1.25
    intraday_breakout = bool(intraday_last is not None and intraday_last > resistance * 1.002 and
                             (intraday_vol_ratio is None or intraday_vol_ratio >= 1.15))

    swing_high, swing_low = float(high.tail(120).max()), float(low.tail(120).min())
    fib = fib_levels(swing_high, swing_low)
    fib_dist = min(abs(price-v)/max(price,1e-9) for v in fib.values())

    # Core scores: S/R gets the heaviest weight for the user's buy-low/sell-high approach.
    ma_score = 100 if golden else 82 if bullish_ma and price_above_50 else 65 if bullish_ma else 25 if price > m200 else 0
    sr_dist = (price-support)/max(price,1e-9)
    sr_score = 100 if sr_dist <= .015 and support_touches >= 3 else 90 if sr_dist <= .025 and support_touches >= 2 else 75 if sr_dist <= .04 else 55 if sr_dist <= .07 else 25
    if price >= resistance * .985 and not breakout_up:
        sr_score = min(sr_score, 45)  # resistance is not a buy-low location

    channel_score = 35
    if near_lower_channel: channel_score += 40
    elif price < center: channel_score += 20
    if slope > 0: channel_score += 15
    if converging: channel_score += 10
    if near_upper_channel: channel_score -= 25
    channel_score = clamp(channel_score)

    breakout_score = 35
    if breakout_volume or intraday_breakout: breakout_score = 100
    elif breakout_up: breakout_score = 80
    elif price >= prior20_high * .98: breakout_score = 70
    elif price >= prior20_high * .94: breakout_score = 55
    elif price > support: breakout_score = 40
    if breakdown: breakout_score = 0

    sec = 0
    sec += 30 if 42 <= rsi_now <= 60 else 20 if 35 <= rsi_now < 42 or 60 < rsi_now <= 67 else 5 if rsi_now > 72 else 12
    sec += 25 if bb_pos <= .30 else 20 if bb_pos <= .45 else 12 if bb_pos <= .70 else 3
    sec += 25 if fib_dist <= .012 else 18 if fib_dist <= .025 else 8
    sec += 20 if macd_hist_now > 0 and macd_hist_now >= macd_hist_prev else 12 if macd_hist_now > 0 else 4

    score = (ma_score * .15 + sr_score * .30 + channel_score * .20 + breakout_score * .15 + sec * .20)
    reasons = []
    if golden: reasons.append("fresh golden cross")
    elif bullish_ma: reasons.append("bullish 50/200 structure")
    else: reasons.append("50/200 structure not bullish")
    if sr_score >= 75: reasons.append(f"repeated support confluence ({support_touches} touches)")
    elif price >= resistance*.985 and not breakout_up: reasons.append("near resistance; poor buy-low location")
    if converging: reasons.append("channel compression/convergence")
    if near_lower_channel: reasons.append("near lower channel boundary")
    if breakout_volume or intraday_breakout: reasons.append("volume-confirmed breakout")
    elif price >= prior20_high*.98: reasons.append("approaching breakout resistance")
    if fib_dist <= .025: reasons.append("near key Fibonacci retracement")
    if bb_pos <= .45: reasons.append("lower-half Bollinger location")
    if macd_hist_now > 0 and macd_hist_now >= macd_hist_prev: reasons.append("improving MACD histogram")
    if rsi_now > 72: reasons.append("overbought penalty")
    if breakdown: reasons.append("20-day breakdown")

    # Buy-low entry: prioritize a pullback to repeated support; otherwise use channel/Fib confluence.
    lower_levels = [(support, support_touches, "repeated support")]
    lower_levels += [(v, c, f"Fib {name}") for name, v in fib.items() if v < price]
    lower_levels.append((ch_lower, 1, "lower channel"))
    viable = [(v,c,n) for v,c,n in lower_levels if v > 0 and v < price and (price-v)/price <= .10]
    if viable:
        # Favor confluence and repeated support, while avoiding an entry materially below a broken level.
        viable.sort(key=lambda x: (-x[1], abs(price-x[0])))
        raw_entry = viable[0][0]
        entry = raw_entry * 1.002
        entry_type = viable[0][2] + " pullback"
    else:
        entry = price * .97
        entry_type = "3% pullback placeholder"

    # If price is already too far above the best support, the scanner says WAIT rather than chase.
    entry_gap = (price-entry)/max(price,1e-9)
    if entry_gap < .005:
        entry = price
        entry_type = "at/near support"

    stop_candidates = [support * .985, entry - 1.75*atr_now]
    valid_stops = [x for x in stop_candidates if np.isfinite(x) and x < entry]
    stop = max(valid_stops) if valid_stops else entry*.93
    risk_per_share = max(entry-stop, 0.01)

    upside_levels = [v for v,c in resistance_candidates if v > entry*1.01]
    upside_levels += [prior20_high, ch_upper]
    target1 = min([v for v in upside_levels if v > entry*1.01], default=entry + 2*risk_per_share)
    target2 = entry + 2*risk_per_share
    target = max(target1, target2)

    capital = float(cfg["capital"])
    shares = capital / entry if entry > 0 else 0
    risk_dollars = risk_per_share * shares
    risk_pct = risk_per_share / entry * 100
    reward_pct = (target-entry)/entry*100
    rr = (target-entry)/risk_per_share if risk_per_share else 0

    major_count = sum([
        ma_score >= 80,
        sr_score >= 75,
        channel_score >= 70,
        breakout_score >= 70,
    ])
    if score >= 85 and major_count >= 3 and entry_gap <= .08:
        state = "HIGH CONVICTION"
    elif score >= 75 and major_count >= 2:
        state = "WATCH / PREPARE LIMIT"
    elif score >= 65:
        state = "DEVELOPING"
    else:
        state = "IGNORE"
    if entry_gap > .08 and score >= 65:
        state = "WAIT FOR PULLBACK"
    if breakdown:
        state = "INVALIDATED"

    return {
        "ticker": ticker, "price": round(price,2), "score": round(score,1), "state": state,
        "entry": round(entry,2), "entry_type": entry_type, "breakout": round(resistance*1.002,2),
        "stop": round(stop,2), "target": round(target,2), "shares": round(shares,4),
        "risk_dollars": round(risk_dollars,2), "risk_pct": round(risk_pct,2), "reward_pct": round(reward_pct,2),
        "rr": round(rr,2), "ma50": round(m50,2), "ma200": round(m200,2),
        "golden": bool(golden), "death": bool(death), "support": round(support,2),
        "resistance": round(resistance,2), "support_touches": int(support_touches),
        "resistance_touches": int(resistance_touches), "rsi": round(rsi_now,1),
        "bb_position": round(bb_pos,2), "bb_width": round(bb_width,3), "atr_pct": round(atr_pct,2),
        "volume_ratio": round(daily_vol_ratio,2), "intraday_volume_ratio": round(intraday_vol_ratio,2) if intraday_vol_ratio else None,
        "converging": bool(converging), "channel_slope_pct": round(slope_pct,3),
        "channel_lower": round(ch_lower,2), "channel_upper": round(ch_upper,2),
        "fib_38": round(fib["38.2"],2), "fib_50": round(fib["50.0"],2), "fib_61": round(fib["61.8"],2),
        "macd_hist": round(macd_hist_now,4), "major_count": int(major_count), "reasons": reasons[:8],
    }


def format_report(results, mode, cfg):
    now = datetime.now(ZoneInfo(cfg["timezone"]))
    title = f"📊 {mode.upper()} STOCK OPPORTUNITY SCAN — {now:%Y-%m-%d %I:%M %p %Z}"
    lines = [title, "", f"Capital model: ${cfg['capital']:.2f}",
             "Core weights: S/R 30% | Channel 20% | MA 15% | Breakout+volume 15% | Secondary 20%",
             "Buy-low rule: prefer repeated support/channel/Fib pullbacks; do not chase extended price.", ""]
    if not results:
        lines += ["⚪ NO QUALIFYING SETUPS", "", "No stock currently meets the minimum technical threshold. Do not force a trade."]
        return "\n".join(lines)
    for i, r in enumerate(results, 1):
        emoji = "🔥" if r["state"] == "HIGH CONVICTION" else "🟢" if r["score"] >= 75 else "🟡"
        lines += [
            f"{emoji} #{i} {r['ticker']} — {r['score']}/100 — {r['state']}",
            f"Price ${r['price']:.2f} | LIMIT ENTRY ${r['entry']:.2f} ({r['entry_type']})",
            f"Breakout confirmation ${r['breakout']:.2f} | Stop ${r['stop']:.2f} | Target ${r['target']:.2f} | R:R {r['rr']:.2f}",
            f"$${cfg['capital']:.2f} model: {r['shares']:.4f} shares | risk ${r['risk_dollars']:.2f} ({r['risk_pct']:.1f}%) | reward {r['reward_pct']:.1f}%",
            "", "OBSERVE",
            f"50MA ${r['ma50']:.2f} | 200MA ${r['ma200']:.2f} | {'fresh golden cross' if r['golden'] else 'fresh death cross' if r['death'] else 'bullish' if r['ma50'] > r['ma200'] else 'bearish'}",
            f"Support ${r['support']:.2f} ({r['support_touches']} touches) | Resistance ${r['resistance']:.2f} ({r['resistance_touches']} touches)",
            f"Channel ${r['channel_lower']:.2f}-${r['channel_upper']:.2f} | {'converging' if r['converging'] else 'not converging'} | slope {r['channel_slope_pct']:.2f}%",
            f"RSI {r['rsi']:.1f} | BB {r['bb_position']:.2f} | ATR {r['atr_pct']:.1f}% | Daily volume {r['volume_ratio']:.2f}x",
            f"Fib 38.2 ${r['fib_38']:.2f} | 50 ${r['fib_50']:.2f} | 61.8 ${r['fib_61']:.2f} | MACD hist {r['macd_hist']:.4f}",
            "", "REASON", "; ".join(r["reasons"]), "", "RESPONSE",
            f"Preferred action: {r['state']}. Set/keep a limit near ${r['entry']:.2f}; do not chase. "
            f"If ${r['breakout']:.2f} breaks with volume, reassess rather than blindly raising the limit.",
            "—" * 34, ""
        ]
    return "\n".join(lines)


def send_discord(content, webhook):
    chunks = []
    while len(content) > 1900:
        cut = content.rfind("\n", 0, 1900)
        if cut < 500: cut = 1900
        chunks.append(content[:cut])
        content = content[cut:].lstrip()
    chunks.append(content)
    for chunk in chunks:
        r = requests.post(webhook, json={"content": chunk}, timeout=20)
        r.raise_for_status()


def should_run(mode, cfg):
    now = datetime.now(ZoneInfo(cfg["timezone"]))
    total = now.hour * 60 + now.minute
    if mode == "intraday":
        return any(abs(total - h*60) <= 20 for h in range(9, 16))
    target = (cfg["morning_local_hour"], cfg["morning_local_minute"]) if mode == "morning" else (cfg["evening_local_hour"], cfg["evening_local_minute"])
    target_total = target[0]*60 + target[1]
    return abs(total-target_total) <= 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["morning", "intraday", "evening"], default="evening")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--force-time", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    is_manual_run = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    if not args.force_time and not is_manual_run and os.getenv("GITHUB_ACTIONS") == "true" and not should_run(args.mode, cfg):
        print("Outside configured local-time window; exiting without sending.")
        return

    tickers = get_universe(cfg)
    print(f"Scanning {len(tickers)} tickers...")
    data = download_prices(tickers, cfg["history_days"], intraday=False)
    intraday_data = download_prices(tickers, cfg["history_days"], intraday=True) if args.mode == "intraday" else None
    results = []
    for ticker in tickers:
        try:
            sub = pd.DataFrame({
                "Open": series_for(data, "Open", ticker), "High": series_for(data, "High", ticker),
                "Low": series_for(data, "Low", ticker), "Close": series_for(data, "Close", ticker),
                "Volume": series_for(data, "Volume", ticker),
            }).dropna(subset=["Close"])
            intraday_sub = None
            if intraday_data is not None:
                intraday_sub = pd.DataFrame({
                    "Open": series_for(intraday_data, "Open", ticker), "High": series_for(intraday_data, "High", ticker),
                    "Low": series_for(intraday_data, "Low", ticker), "Close": series_for(intraday_data, "Close", ticker),
                    "Volume": series_for(intraday_data, "Volume", ticker),
                }).dropna(subset=["Close"])
            if len(sub) >= 220:
                r = analyze_ticker(ticker, sub, cfg, intraday_sub)
                if r and r["score"] >= cfg["min_score_to_report"]:
                    results.append(r)
        except Exception as e:
            print(f"{ticker}: {e}")

    results.sort(key=lambda x: (x["score"], x["major_count"], x["rr"], x["support_touches"]), reverse=True)
    results = results[:cfg["max_candidates"]]
    report = format_report(results, args.mode, cfg)
    print(report)
    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(reports_dir, f"{stamp}_{args.mode}.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    if args.send:
        webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook:
            print("DISCORD_WEBHOOK_URL missing; report printed but not sent.")
            return
        send_discord(report, webhook)


if __name__ == "__main__":
    main()
