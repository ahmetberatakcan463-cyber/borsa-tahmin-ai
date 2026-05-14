import os
import requests
import warnings
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import ta
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

warnings.filterwarnings("ignore")

app = Flask(__name__)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PORT = int(os.environ.get("PORT", 5050))

prediction_history = {}

SECTORS = {
    "Bankacilik": ["GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "TSKB.IS"],
    "Havacilik": ["THYAO.IS", "PGSUS.IS"],
    "Enerji": ["TUPRS.IS", "PETKM.IS", "ENKAI.IS"],
    "Savunma": ["ASELS.IS"],
    "Telecom": ["TCELL.IS", "TTKOM.IS"],
    "Otomotiv": ["FROTO.IS", "TOASO.IS", "TTRAK.IS", "ARCLK.IS", "BRISA.IS"],
    "Perakende": ["BIMAS.IS", "MGROS.IS", "SOKM.IS", "ULKER.IS", "AEFES.IS"],
    "Holding": ["KCHOL.IS", "SAHOL.IS", "DOHOL.IS", "ALARK.IS", "NTHOL.IS"],
    "Sanayi": ["EREGL.IS", "SASA.IS", "SISE.IS", "GUBRF.IS"],
    "GYO": ["EKGYO.IS"],
    "Madencilik": ["KOZAL.IS"],
    "Teknoloji": ["LOGO.IS", "INDES.IS"],
    "Diger": ["MAVI.IS", "HEKTS.IS", "VESBE.IS", "BUCIM.IS", "CIMSA.IS",
               "OYAKC.IS", "PRKAB.IS", "EGEEN.IS", "CANTE.IS", "KERVT.IS", "GESAN.IS", "KONTR.IS"]
}

BIST_STOCKS = list({s for stocks in SECTORS.values() for s in stocks})


def get_ticker_sector(ticker):
    for sector, stocks in SECTORS.items():
        if ticker in stocks:
            return sector
    return "Diger"


def get_news_sentiment(ticker_name):
    try:
        query = ticker_name + "+hisse+borsa"
        url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
        feed = feedparser.parse(url)
        pos_words = ["yukselis", "artis", "kazanc", "rekor", "tavan", "guclu",
                     "pozitif", "alim", "buyume", "kar", "atladi", "firladi", "yukseldi"]
        neg_words = ["dusus", "kayip", "zarar", "satis", "negatif", "baski",
                     "risk", "endise", "geriledi", "dustu", "coktu", "erid"]
        pos, neg = 0, 0
        titles = []
        for entry in feed.entries[:6]:
            title = entry.title if hasattr(entry, "title") else ""
            titles.append(title)
            tl = title.lower()
            for w in pos_words:
                if w in tl: pos += 1
            for w in neg_words:
                if w in tl: neg += 1
        sentiment = "pozitif" if pos > neg else ("negatif" if neg > pos else "nottr")
        return {"positive": pos, "negative": neg, "titles": titles[:3], "sentiment": sentiment}
    except Exception:
        return {"positive": 0, "negative": 0, "titles": [], "sentiment": "nottr"}


def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        close_s = pd.Series(close.values, dtype=float)
        volume_s = pd.Series(volume.values, dtype=float)
        high_s = pd.Series(high.values, dtype=float)
        low_s = pd.Series(low.values, dtype=float)

        # RSI
        rsi_ind = ta.momentum.RSIIndicator(close=close_s, window=14)
        rsi_val = float(rsi_ind.rsi().iloc[-1])

        # MACD
        macd_ind = ta.trend.MACD(close=close_s)
        macd_hist_val = float(macd_ind.macd_diff().iloc[-1])
        macd_bull = macd_hist_val > 0

        # OBV
        obv_ind = ta.volume.OnBalanceVolumeIndicator(close=close_s, volume=volume_s)
        obv_s = obv_ind.on_balance_volume()
        obv_bull = float(obv_s.iloc[-1]) > float(obv_s.iloc[-10]) if len(obv_s) >= 10 else False

        # Bollinger Bands
        bb_ind = ta.volatility.BollingerBands(close=close_s, window=20, window_dev=2)
        bbl = float(bb_ind.bollinger_lband().iloc[-1])
        bbu = float(bb_ind.bollinger_hband().iloc[-1])
        cur_price = float(close_s.iloc[-1])
        bb_pos = (cur_price - bbl) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5

        # Stochastic
        stoch_ind = ta.momentum.StochasticOscillator(high=high_s, low=low_s, close=close_s, window=14, smooth_window=3)
        stoch_k = float(stoch_ind.stoch().iloc[-1])

        # 52-week
        w52_high = float(high_s.max())
        w52_low = float(low_s.min())
        pct_from_high = (cur_price / w52_high - 1) * 100

        # Pivot / Support / Resistance
        r_high = float(high_s.tail(20).max())
        r_low = float(low_s.tail(20).min())
        pivot = (r_high + r_low + cur_price) / 3
        resistance1 = round(2 * pivot - r_low, 2)
        support1 = round(2 * pivot - r_high, 2)

        # Volume
        avg_vol = float(volume_s.tail(20).mean())
        last_vol = float(volume_s.iloc[-1])
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        # Momentum
        mom_5 = float((close_s.iloc[-1] / close_s.iloc[-6] - 1) * 100) if len(close_s) >= 6 else 0
        mom_20 = float((close_s.iloc[-1] / close_s.iloc[-21] - 1) * 100) if len(close_s) >= 21 else 0
        daily_change = float((close_s.iloc[-1] / close_s.iloc[-2] - 1) * 100) if len(close_s) >= 2 else 0

        # SCORE
        score = 0
        if 58 <= rsi_val <= 72:   score += 18
        elif 50 <= rsi_val < 58:  score += 9
        elif rsi_val > 72:        score += 3
        if macd_bull:
            score += 14
            if macd_hist_val > 0.5: score += 4
        if obv_bull: score += 14
        if vol_ratio >= 3.0:   score += 20
        elif vol_ratio >= 2.0: score += 14
        elif vol_ratio >= 1.5: score += 8
        elif vol_ratio >= 1.2: score += 4
        if pct_from_high >= 0:         score += 18
        elif -3 <= pct_from_high < 0:  score += 14
        elif -8 <= pct_from_high < -3: score += 7
        if mom_5 > 7:   score += 8
        elif mom_5 > 3: score += 5
        elif mom_5 > 0: score += 2
        if 0.65 <= bb_pos <= 0.92: score += 8
        elif 0.5 <= bb_pos < 0.65: score += 3
        if 50 <= stoch_k <= 80: score += 4

        sector = get_ticker_sector(ticker)
        return {
            "ticker": ticker.replace(".IS", ""),
            "full_ticker": ticker,
            "sector": sector,
            "price": round(cur_price, 2),
            "daily_change": round(daily_change, 2),
            "rsi": round(rsi_val, 1),
            "macd_bullish": macd_bull,
            "macd_hist": round(macd_hist_val, 3),
            "obv_bullish": obv_bull,
            "stoch_k": round(stoch_k, 1),
            "volume_ratio": round(vol_ratio, 2),
            "momentum_5d": round(mom_5, 2),
            "momentum_20d": round(mom_20, 2),
            "w52_high": round(w52_high, 2),
            "w52_low": round(w52_low, 2),
            "pct_from_high": round(pct_from_high, 2),
            "resistance1": resistance1,
            "support1": support1,
            "bb_pos": round(bb_pos, 2),
            "score": min(score, 100),
            "news": None,
            "sector_momentum": 0.0
        }
    except Exception:
        return None


def get_crypto_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 30,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d"
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        result = []
        for coin in resp.json():
            c24 = coin.get("price_change_percentage_24h") or 0
            c7d = coin.get("price_change_percentage_7d_in_currency") or 0
            c1h = coin.get("price_change_percentage_1h_in_currency") or 0
            vol = coin.get("total_volume") or 0
            mcap = coin.get("market_cap") or 1
            vtm = (vol / mcap) * 100
            score = 0
            if c24 > 5:    score += 30
            elif c24 > 2:  score += 20
            elif c24 > 0:  score += 10
            if c7d > 10:   score += 25
            elif c7d > 5:  score += 15
            elif c7d > 0:  score += 8
            if vtm > 20:   score += 25
            elif vtm > 10: score += 15
            elif vtm > 5:  score += 8
            if c1h > 1:    score += 20
            elif c1h > 0:  score += 10
            result.append({
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "price": coin.get("current_price"),
                "change_1h": round(c1h, 2),
                "change_24h": round(c24, 2),
                "change_7d": round(c7d, 2),
                "volume": vol,
                "vol_to_mcap": round(vtm, 2),
                "score": min(score, 100),
                "image": coin.get("image")
            })
        return sorted(result, key=lambda x: x["score"], reverse=True)
    except Exception:
        return []


def ask_claude(prompt, system=""):
    if not ANTHROPIC_API_KEY:
        return "ANTHROPIC_API_KEY eksik."
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    today = datetime.now().strftime("%d.%m.%Y")
    default_system = (
        f"Sen Türkiye borsası (BIST), kripto para ve global piyasalar konusunda uzman cesur bir finans analistisin. Bugün: {today}. "
        "KESIN KURAL: 'kesin söyleyemem', 'garanti edemem', 'profesyonel danışmana gidin', 'yatırım tavsiyesi değildir' gibi ifadeler KULLANMA. "
        "Her soruda mutlaka somut rakam ver: fiyat hedefi, tarih, yüzde tahmin. "
        "Yanıtın en sonuna sadece küçük '(Bu bir tahmindir)' yaz. Türkçe yaz."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1800,
        system=system or default_system,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def save_prediction(top_stocks):
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_history[today] = [
        {"ticker": s["ticker"], "price": s["price"], "score": s["score"]}
        for s in top_stocks
    ]


def get_accuracy():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if yesterday not in prediction_history:
        return None
    past = prediction_history[yesterday]
    correct, results = 0, []
    for p in past:
        try:
            df = yf.download(p["ticker"] + ".IS", period="2d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 2:
                continue
            close = df["Close"].squeeze()
            cur = float(close.iloc[-1])
            change = (cur - p["price"]) / p["price"] * 100
            if change > 0: correct += 1
            results.append({
                "ticker": p["ticker"],
                "predicted_price": p["price"],
                "actual_price": round(cur, 2),
                "change": round(change, 2),
                "hit_tavan": change >= 9.5
            })
        except Exception:
            pass
    acc = (correct / len(results) * 100) if results else 0
    return {"accuracy": round(acc, 1), "results": results, "date": yesterday}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tavan", methods=["GET"])
def api_tavan():
    all_results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(analyze_stock, t): t for t in BIST_STOCKS}
        for f in as_completed(futures):
            r = f.result()
            if r:
                all_results.append(r)

    sector_moms = defaultdict(list)
    for r in all_results:
        sector_moms[r["sector"]].append(r["momentum_5d"])
    sector_avg = {s: sum(v) / len(v) for s, v in sector_moms.items() if v}

    for r in all_results:
        avg = sector_avg.get(r["sector"], 0)
        r["sector_momentum"] = round(avg, 2)
        if avg > 5:    r["score"] = min(r["score"] + 10, 100)
        elif avg > 2:  r["score"] = min(r["score"] + 5, 100)
        elif avg < -3: r["score"] = max(r["score"] - 5, 0)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top = all_results[:10]

    with ThreadPoolExecutor(max_workers=5) as ex:
        news_futures = {ex.submit(get_news_sentiment, s["ticker"]): i for i, s in enumerate(top)}
        for f in as_completed(news_futures):
            idx = news_futures[f]
            news = f.result()
            top[idx]["news"] = news
            if news["sentiment"] == "pozitif":
                top[idx]["score"] = min(top[idx]["score"] + 10, 100)
            elif news["sentiment"] == "negatif":
                top[idx]["score"] = max(top[idx]["score"] - 5, 0)

    top.sort(key=lambda x: x["score"], reverse=True)
    save_prediction(top)

    yarin = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    summary = "\n".join([
        f"- {s['ticker']} [{s['sector']}]: {s['price']} TL | Günlük={s['daily_change']}% | RSI={s['rsi']} | "
        f"MACD={'yukari' if s['macd_bullish'] else 'asagi'} | OBV={'yukari' if s['obv_bullish'] else 'asagi'} | "
        f"Hacim={s['volume_ratio']}x | 5g Mom={s['momentum_5d']}% | 52hZirve={s['pct_from_high']}% | "
        f"Direnc={s['resistance1']} TL | Destek={s['support1']} TL | Haber={s.get('news', {}).get('sentiment','notr')} | Skor={s['score']}/100"
        for s in top
    ])

    prompt = f"""Bugun ({datetime.now().strftime('%d.%m.%Y')}) BIST'de yaptigim gelismis teknik analiz sonuclari:

{summary}

Asagidaki formatta KESIN tahmin ver:

YARIN ({yarin}) TAVAN ADAYLARI:
Her hisse icin:
▸ Hisse: [isim] | Sektor: [sektor]
▸ Bugunku fiyat: X TL → Yarin hedef: Y TL (tavan %10 = Z TL)
▸ Tavan ihtimali: %XX
▸ Guclu sinyaller: [RSI/OBV/Hacim/Haber gerekce]
▸ Risk: [kisa uyari]

En az 4 hisse ver. Sonunda sektor bazli genel yorum + BIST 100 endeks hedefi yaz."""

    ai_analysis = ask_claude(prompt)
    return jsonify({
        "stocks": all_results[:25],
        "top_candidates": top,
        "ai_analysis": ai_analysis,
        "sector_avg": sector_avg,
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M")
    })


@app.route("/api/crypto", methods=["GET"])
def api_crypto():
    coins = get_crypto_data()
    top = coins[:10]
    yarin = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    summary = "\n".join([
        f"- {c['name']} ({c['symbol']}): ${c['price']} | 1s={c['change_1h']}% | "
        f"24s={c['change_24h']}% | 7g={c['change_7d']}% | Hacim/Mcap={c['vol_to_mcap']}% | Skor={c['score']}/100"
        for c in top
    ])
    prompt = f"""Bugun ({datetime.now().strftime('%d.%m.%Y')}) kripto para piyasasi verileri:

{summary}

Asagidaki formatta KESIN tahmin ver:

24-48 SAAT KRIPTO TAHMINI ({yarin}):
Her coin icin:
▸ Coin: [isim] ([sembol])
▸ Simdiki fiyat: $X → 48 saat hedef: $Y
▸ Yukselis ihtimali: %XX
▸ Guclu sinyaller: [gerekcе]

Bitcoin icin mutlaka ayri bolum: simdiki fiyat, 48 saat hedef, 1 hafta hedef.
En az 4 coin ver. Sonunda genel kripto piyasasi yorumu yap."""

    ai_analysis = ask_claude(prompt)
    return jsonify({
        "coins": coins,
        "top_candidates": top,
        "ai_analysis": ai_analysis,
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M")
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Soru bos olamaz"}), 400
    today = datetime.now().strftime("%d.%m.%Y")
    system = (
        f"Sen BIST, kripto, forex ve global piyasalar konusunda deneyimli cesur bir finans analistisin. Bugun: {today}. "
        "KESIN KURAL: Asla 'kesin soyleyemem', 'garanti edemem', 'danismana gidin', 'yatirim tavsiyesi degildir' deme. "
        "Her soruda mutlaka somut fiyat hedefi, tarih ve yuzde tahmin ver. "
        "Yanıtın en sonuna sadece '(Bu bir tahmindir)' ekle. Turkce yaz."
    )
    answer = ask_claude(question, system=system)
    return jsonify({"answer": answer, "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M")})


@app.route("/api/accuracy", methods=["GET"])
def api_accuracy():
    result = get_accuracy()
    if not result:
        return jsonify({"message": "Dun tahmin kaydi yok."})
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 60)
    print("  BORSA TAHMIN AI v2 - Basliyor...")
    print(f"  API Key: {'Mevcut' if ANTHROPIC_API_KEY else 'EKSIK!'}")
    print(f"  Adres: http://localhost:{PORT}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=PORT)
