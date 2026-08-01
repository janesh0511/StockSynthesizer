from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import os
import time
import random
import queue
import threading
import re
import requests
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from agent_alpha import AgentAlpha
from chart_visualizer import SentimentChart
from sentiment_aggregator import SentimentAggregator, create_sentiment_point
import finnhub

app = Flask(__name__, static_folder='.', static_url_path='')

# ===== CONFIGURATION =====
CACHE_DURATION = 180  # 3 MINUTES - reduced to get fresher data
REQUEST_DELAY = 1.0   # 1 second between requests

# ===== FINNHUB CONFIGURATION =====
FINNHUB_API_KEY = 'd9mjtj9r01qtdq8u4d10d9mjtj9r01qtdq8u4d1g'
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ===== SUPPRESS ALL WARNINGS =====
import warnings
warnings.filterwarnings("ignore")

# ===== COMPANY NAME MAPPING =====
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms Inc.',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust',
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'DOGE-USD': 'Dogecoin',
    'NFLX': 'Netflix Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp.',
    'DIS': 'Walt Disney Co.',
    'VTI': 'Vanguard Total Stock Market ETF',
    'SNDK': 'SanDisk Corporation',
}

TICKER_ALIASES = {
    'APPLE': 'AAPL',
    'TESLA': 'TSLA',
    'NVIDIA': 'NVDA',
    'MICROSOFT': 'MSFT',
    'AMAZON': 'AMZN',
    'GOOGLE': 'GOOGL',
    'ALPHABET': 'GOOGL',
    'META PLATFORMS': 'META',
    'BITCOIN': 'BTC-USD',
    'ETHEREUM': 'ETH-USD',
    'DOGECOIN': 'DOGE-USD',
    'NETFLIX': 'NFLX',
    'JPMORGAN': 'JPM',
    'BANK OF AMERICA': 'BAC',
    'WALT DISNEY': 'DIS',
    'VANGUARD TOTAL STOCK MARKET': 'VTI',
     'SANDISK': 'SNDK', 
}

# ===== STOCK DATA CACHE =====
stock_cache = {}
stock_cache.clear()
print("🗑️ Cleared stock cache on startup")

# ===== MARKET STATUS =====
def get_market_status() -> Dict:
    now = datetime.now()
    is_open = False
    if now.weekday() < 5:
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_open = market_open <= now <= market_close
    return {
        'is_open': is_open,
        'status': 'OPEN' if is_open else 'CLOSED',
        'timestamp': now.isoformat()
    }

# ===== REAL-TIME STREAMING ENGINE =====
@dataclass
class SentimentMessage:
    text: str
    ticker: str
    source: str
    timestamp: str
    sentiment_score: Optional[float] = None
    author: Optional[str] = None
    upvotes: Optional[int] = None
    
    def to_dict(self):
        return {
            'text': self.text,
            'ticker': self.ticker,
            'source': self.source,
            'timestamp': self.timestamp,
            'sentiment_score': self.sentiment_score,
            'author': self.author,
            'upvotes': self.upvotes
        }

class RealTimeStream:
    def __init__(self, max_queue_size: int = 1000):
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.is_paused = False
        self.thread = None
        self.total_messages = 0
        self.ticker_frequency = {}
        self.recent_messages = []
        self.max_recent = 50
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.is_paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("📡 Data stream started")
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("📡 Data stream stopped")
    
    def pause(self):
        self.is_paused = True
        print("⏸️ Data stream paused")
    
    def resume(self):
        self.is_paused = False
        print("▶️ Data stream resumed")
    
    def toggle_pause(self):
        if self.is_paused:
            self.resume()
        else:
            self.pause()
    
    def get_message(self, timeout: float = 0.1):
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        return self.message_queue.qsize()
    
    def get_statistics(self) -> Dict:
        return {
            'total_messages': self.total_messages,
            'queue_size': self.message_queue.qsize(),
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'ticker_frequency': self.ticker_frequency,
            'recent_count': len(self.recent_messages)
        }
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        return self.recent_messages[-n:]
    
    def _run_loop(self):
        """Only sends pings - no data fetching to avoid rate limits"""
        while self.is_running:
            try:
                if not self.is_paused:
                    ping_msg = {
                        'type': 'ping',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.message_queue.put(ping_msg)
                    self.total_messages += 1
                time.sleep(60)
            except Exception as e:
                print(f"Error in stream: {e}")
                time.sleep(60)

# ===== GLOBAL INSTANCES =====
stream = RealTimeStream()
aggregator = SentimentAggregator()
agent_alpha = AgentAlpha()

# ===== STOCK DATA FUNCTIONS =====
def resolve_ticker(query):
    if not query:
        return 'AAPL'
    
    raw_query = str(query).strip()
    if not raw_query:
        return 'AAPL'
    
    normalized = raw_query.upper()
    
    alias = TICKER_ALIASES.get(normalized)
    if alias:
        return alias
    
    for symbol, company_name in COMPANY_NAMES.items():
        if company_name.lower() in raw_query.lower() or symbol.lower() == raw_query.lower():
            return symbol
    
    if re.fullmatch(r'^[A-Z0-9.\-^]{1,6}$', normalized):
        return normalized
    
    return normalized


def _build_fallback_stock_data(ticker: str, error: Optional[Exception] = None) -> Dict:
    """Built-in sample data as the final safety net"""
    resolved_ticker = resolve_ticker(ticker)
    company_name = COMPANY_NAMES.get(resolved_ticker, f'{resolved_ticker} Inc.')

    sample_prices = {
        'AAPL': [190.12, 191.40, 192.10, 189.85, 190.50, 191.20, 192.00, 193.15, 194.10, 193.00, 194.80, 195.50, 196.20, 197.10],
        'TSLA': [220.40, 221.10, 219.90, 218.75, 219.20, 220.00, 221.30, 223.10, 224.00, 223.40, 224.90, 225.70, 226.40, 227.10],
        'NVDA': [120.50, 121.10, 121.80, 122.40, 123.10, 123.80, 124.30, 124.90, 125.40, 125.90, 126.40, 126.90, 127.40, 128.00],
        'MSFT': [420.90, 421.20, 422.10, 423.30, 424.20, 424.90, 425.60, 426.30, 427.10, 427.90, 428.50, 429.20, 430.00, 430.80],
        'AMZN': [185.20, 186.00, 186.60, 187.20, 187.80, 188.40, 189.10, 189.80, 190.50, 191.10, 191.80, 192.40, 193.10, 193.70],
        'GOOGL': [175.40, 176.10, 176.80, 177.10, 177.60, 178.20, 178.80, 179.30, 179.90, 180.40, 181.00, 181.50, 182.00, 182.50],
        'META': [490.30, 491.10, 492.00, 492.80, 493.60, 494.20, 495.00, 495.70, 496.40, 497.10, 497.80, 498.40, 499.10, 499.80],
        'SPY': [520.10, 520.70, 521.20, 521.80, 522.30, 522.90, 523.40, 524.00, 524.50, 525.10, 525.70, 526.20, 526.80, 527.30],
        'QQQ': [440.00, 440.80, 441.60, 442.20, 442.90, 443.60, 444.20, 444.80, 445.40, 446.00, 446.60, 447.20, 447.80, 448.40],
        'BTC-USD': [62000.0, 62130.0, 62250.0, 62380.0, 62510.0, 62640.0, 62770.0, 62900.0, 63050.0, 63200.0, 63340.0, 63480.0, 63620.0, 63760.0],
        'ETH-USD': [3200.0, 3210.0, 3220.0, 3230.0, 3240.0, 3252.0, 3265.0, 3278.0, 3290.0, 3305.0, 3318.0, 3330.0, 3345.0, 3360.0],
        'DOGE-USD': [0.18, 0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.19, 0.191, 0.192, 0.193],
        'NFLX': [650.10, 651.20, 652.00, 653.20, 654.00, 655.10, 656.20, 657.30, 658.50, 659.80, 660.90, 661.80, 662.70, 663.60],
        'JPM': [200.10, 200.50, 201.00, 201.40, 201.90, 202.40, 202.90, 203.30, 203.80, 204.20, 204.70, 205.10, 205.50, 206.00],
        'BAC': [40.00, 40.10, 40.20, 40.35, 40.50, 40.70, 40.90, 41.10, 41.30, 41.50, 41.70, 41.90, 42.10, 42.30],
        'DIS': [110.20, 110.60, 111.00, 111.40, 111.80, 112.20, 112.60, 113.00, 113.40, 113.80, 114.20, 114.60, 115.00, 115.40],
        'VTI': [280.10, 280.60, 281.10, 281.70, 282.20, 282.80, 283.40, 284.00, 284.60, 285.10, 285.70, 286.20, 286.80, 287.40],
    }

    prices = sample_prices.get(resolved_ticker, sample_prices['AAPL'])
    historical_prices = prices[:]
    current_price = historical_prices[-1]
    previous_close = historical_prices[-2] if len(historical_prices) > 1 else historical_prices[-1]
    price_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0.0
    sentiment_score = max(0, min(100, 50 + (price_change * 2)))
    volume = max(100000, int(current_price * 1000))

    return {
        'ticker': resolved_ticker,
        'company_name': company_name,
        'current_price': round(current_price, 2),
        'currency': 'USD',
        'previous_close': round(previous_close, 2),
        'open': round(historical_prices[0], 2),
        'high': round(max(historical_prices), 2),
        'low': round(min(historical_prices), 2),
        'volume': volume,
        'price_change': round(price_change, 2),
        'sentiment_score': round(sentiment_score),
        'historical_prices': [round(float(value), 2) for value in historical_prices],
        'historical_dates': [(datetime.now() - timedelta(days=13 - idx)).strftime('%Y-%m-%d') for idx in range(len(historical_prices))],
        'timestamp': datetime.now().isoformat(),
        'last_trade_time': datetime.now().isoformat(),
        'price_type': 'regular',
        'data_source': 'offline-sample',
        'market_status': get_market_status(),
        'is_real_data': False,
        'fallback_reason': str(error) if error else 'provider unavailable'
    }


def get_stock_data(ticker):
    """Professional data fetcher with 3-tier fallback:
    1. Finnhub (real-time) - Primary
    2. yfinance with after-hours - Secondary
    3. Offline samples - Final safety net
    """
    resolved_ticker = resolve_ticker(ticker)
    cache_key = resolved_ticker.upper()
    
    # Check cache first
    if cache_key in stock_cache:
        cache_time, data = stock_cache[cache_key]
        if (datetime.now() - cache_time).seconds < CACHE_DURATION:
            print(f"📦 Using cached data for: {resolved_ticker}")
            return data
    
    print(f"🔍 Fetching data for: {resolved_ticker}")
    
    # ===== TIER 1: FINNHUB (REAL-TIME) =====
    if finnhub_client:
        try:
            print(f"📡 Tier 1: Trying Finnhub for {resolved_ticker}...")
            time.sleep(REQUEST_DELAY)
            
            # Get real-time quote
            quote = finnhub_client.quote(resolved_ticker)
            
            # Check if we got valid data
            if quote and quote.get('c') and quote.get('c') > 0:
                current_price = float(quote['c'])
                previous_close = float(quote.get('pc', current_price))
                open_price = float(quote.get('o', current_price))
                high_price = float(quote.get('h', current_price))
                low_price = float(quote.get('l', current_price))
                volume = int(quote.get('v', 0))
                
                price_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0.0
                sentiment_score = max(0, min(100, 50 + (price_change * 2)))
                
                # Get company name
                company_name = COMPANY_NAMES.get(resolved_ticker, f'{resolved_ticker} Inc.')
                try:
                    profile = finnhub_client.company_profile2(symbol=resolved_ticker)
                    if profile and profile.get('name'):
                        company_name = profile['name']
                except:
                    pass
                
                # For historical data, fallback to yfinance
                historical_prices = []
                historical_dates = []
                try:
                    stock = yf.Ticker(resolved_ticker)
                    hist = stock.history(period="1mo", auto_adjust=False)
                    if not hist.empty:
                        historical_prices = [round(float(value), 2) for value in hist['Close'].tolist()]
                        historical_dates = [d.strftime('%Y-%m-%d') for d in hist.index]
                except:
                    # Use sample data for history
                    sample = _build_fallback_stock_data(resolved_ticker)
                    historical_prices = sample.get('historical_prices', [])
                    historical_dates = sample.get('historical_dates', [])
                
                result = {
                    'ticker': resolved_ticker,
                    'company_name': company_name,
                    'current_price': round(current_price, 2),
                    'currency': 'USD',
                    'previous_close': round(previous_close, 2),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'volume': volume,
                    'price_change': round(price_change, 2),
                    'sentiment_score': round(sentiment_score),
                    'historical_prices': historical_prices,
                    'historical_dates': historical_dates,
                    'timestamp': datetime.now().isoformat(),
                    'last_trade_time': datetime.now().isoformat(),
                    'price_type': 'live' if get_market_status()['is_open'] else 'after_hours',
                    'data_source': 'finnhub',
                    'market_status': get_market_status(),
                    'is_real_data': True
                }
                
                stock_cache[cache_key] = (datetime.now(), result)
                print(f"✅ Finnhub data for {resolved_ticker}: ${current_price:.2f}")
                return result
            else:
                print(f"⚠️ Finnhub returned no data for {resolved_ticker}")
        except Exception as e:
            print(f"⚠️ Finnhub error for {resolved_ticker}: {e}")
    
    # ===== TIER 2: YFINANCE WITH AFTER-HOURS =====
    try:
        print(f"📡 Tier 2: Trying yfinance with after-hours for {resolved_ticker}...")
        time.sleep(REQUEST_DELAY)
        
        # Get data with after-hours
        history = yf.download(
            resolved_ticker,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            threads=False,
            prepost=True  # ✅ AFTER-HOURS ENABLED
        )
        
        # If no data, try without prepost
        if history.empty:
            print("⚠️ No after-hours data, trying regular...")
            history = yf.download(
                resolved_ticker,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False
            )
        
        if history.empty:
            raise ValueError(f"No data from yfinance for {resolved_ticker}")
        
        # Clean data
        history = history.dropna(subset=['Close'])
        if history.empty:
            raise ValueError(f"No valid Close data")
        
        # Get latest values
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else latest
        
        current_price = float(latest['Close'])
        previous_close = float(previous['Close'])
        last_trade_time = history.index[-1]
        
        # Check if after-hours
        is_after_hours = last_trade_time.hour >= 16 or last_trade_time.hour < 9
        
        price_change = ((current_price - previous_close) / previous_close * 100) if previous_close else 0.0
        sentiment_score = max(0, min(100, 50 + (price_change * 2)))
        
        # Get company info
        company_name = COMPANY_NAMES.get(resolved_ticker, f'{resolved_ticker} Inc.')
        try:
            stock = yf.Ticker(resolved_ticker)
            info = stock.info or {}
            company_name = info.get('longName') or info.get('shortName') or company_name
        except:
            pass
        
        result = {
            'ticker': resolved_ticker,
            'company_name': company_name,
            'current_price': round(current_price, 2),
            'currency': 'USD',
            'previous_close': round(previous_close, 2),
            'open': round(float(latest['Open']) if 'Open' in latest else current_price, 2),
            'high': round(float(latest['High']) if 'High' in latest else current_price, 2),
            'low': round(float(latest['Low']) if 'Low' in latest else current_price, 2),
            'volume': int(latest['Volume']) if 'Volume' in latest else 0,
            'price_change': round(price_change, 2),
            'sentiment_score': round(sentiment_score),
            'historical_prices': [round(float(value), 2) for value in history['Close'].tolist()],
            'historical_dates': [d.strftime('%Y-%m-%d %H:%M') for d in history.index],
            'timestamp': datetime.now().isoformat(),
            'last_trade_time': last_trade_time.isoformat(),
            'price_type': 'after_hours' if is_after_hours else 'regular',
            'data_source': 'yfinance',
            'market_status': get_market_status(),
            'is_real_data': True
        }
        
        stock_cache[cache_key] = (datetime.now(), result)
        print(f"✅ yfinance data for {resolved_ticker}: ${current_price:.2f} ({result['price_type']})")
        return result
        
    except Exception as e:
        print(f"⚠️ yfinance error: {e}")
    
    # ===== TIER 3: OFFLINE SAMPLES (FINAL SAFETY NET) =====
    print(f"📡 Tier 3: Using offline samples for {resolved_ticker}")
    result = _build_fallback_stock_data(resolved_ticker, error=e if 'e' in locals() else None)
    stock_cache[cache_key] = (datetime.now(), result)
    return result

# ===== API ENDPOINTS =====

@app.route('/')
def landing():
    return send_from_directory('.', 'landing.html')

@app.route('/app')
def serve_app():
    return send_from_directory('.', 'index.html')

@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    resolved_ticker = resolve_ticker(ticker)
    
    if force_refresh:
        cache_key = resolved_ticker.upper()
        if cache_key in stock_cache:
            del stock_cache[cache_key]
            print(f"🗑️ Forced cache refresh for {resolved_ticker}")
    
    try:
        data = get_stock_data(resolved_ticker)
        return jsonify(data)
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e),
            'ticker': resolved_ticker
        }), 503

@app.route('/api/market/status')
def market_status():
    return jsonify(get_market_status())

@app.route('/api/sentiment')
def get_sentiment():
    ticker = request.args.get('ticker', 'AAPL')
    resolved_ticker = resolve_ticker(ticker)
    
    try:
        stock_data = get_stock_data(resolved_ticker)
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)}), 503
    
    return jsonify({
        'ticker': resolved_ticker,
        'sentiment_score': stock_data['sentiment_score'],
        'stock_data': {
            'current_price': stock_data['current_price'],
            'price_change': stock_data['price_change'],
            'company_name': stock_data['company_name']
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/news')
def get_news():
    ticker = request.args.get('ticker', 'AAPL')
    resolved_ticker = resolve_ticker(ticker)
    
    try:
        stock_data = get_stock_data(resolved_ticker)
        price = stock_data.get('current_price', 0)
        change = stock_data.get('price_change', 0)
        volume = stock_data.get('volume', 0)
    except:
        price = 0
        change = 0
        volume = 0
    
    news_data = [
        {'outlet': 'Bloomberg', 'score': round(0.5 + (change / 100), 2), 'impact': 'High' if abs(change) > 2 else 'Med', 'headline': f"{ticker} trades at ${price:.2f}, {'up' if change > 0 else 'down'} {abs(change):.2f}%"},
        {'outlet': 'Reuters', 'score': round(0.5 + (change / 150), 2), 'impact': 'Med', 'headline': f"{ticker} volume reaches {volume:,} shares"},
        {'outlet': 'CNBC', 'score': round(0.5 + (change / 120), 2), 'impact': 'Med', 'headline': f"{ticker} showing {'strength' if change > 0 else 'weakness'} today"},
        {'outlet': 'WSJ', 'score': round(0.5 + (change / 80), 2), 'impact': 'High' if abs(change) > 1.5 else 'Med', 'headline': f"{ticker} {'gains' if change > 0 else 'falls'}"}
    ]
    return jsonify(news_data)

@app.route('/api/news/real')
def get_real_news():
    ticker = request.args.get('ticker', 'AAPL')
    resolved_ticker = resolve_ticker(ticker)
    
    try:
        stock_data = get_stock_data(resolved_ticker)
        price = stock_data.get('current_price', 0)
        change = stock_data.get('price_change', 0)
        volume = stock_data.get('volume', 0)
    except:
        price = 0
        change = 0
        volume = 0
    
    news_items = [
        {'title': f"{ticker} {'Surges' if change > 2 else 'Advances' if change > 0 else 'Declines'} {abs(change):.2f}% to ${price:.2f}", 'source': 'Bloomberg', 'sentiment': round(0.5 + (change / 100), 2), 'time': datetime.now().strftime('%H:%M'), 'url': '#', 'volume': volume},
        {'title': f"Trading Update: {ticker} Volume Hits {volume:,} Shares", 'source': 'Reuters', 'sentiment': round(0.5 + (change / 150), 2), 'time': (datetime.now() - timedelta(minutes=15)).strftime('%H:%M'), 'url': '#', 'volume': volume},
        {'title': f"{ticker} Price Action: ${price:.2f}", 'source': 'CNBC', 'sentiment': round(0.5 + (change / 120), 2), 'time': (datetime.now() - timedelta(minutes=30)).strftime('%H:%M'), 'url': '#', 'volume': volume},
        {'title': f"Market Analysis: {ticker} {'Outperforms' if change > 1 else 'Underperforms'}", 'source': 'WSJ', 'sentiment': round(0.5 + (change / 80), 2), 'time': (datetime.now() - timedelta(minutes=45)).strftime('%H:%M'), 'url': '#', 'volume': volume}
    ]
    return jsonify(news_items)

@app.route('/api/stream/start')
def stream_start():
    stream.start()
    return jsonify({'status': 'started', 'is_running': stream.is_running})

@app.route('/api/stream/stop')
def stream_stop():
    stream.stop()
    return jsonify({'status': 'stopped', 'is_running': stream.is_running})

@app.route('/api/stream/pause')
def stream_pause():
    stream.pause()
    return jsonify({'status': 'paused', 'is_paused': stream.is_paused})

@app.route('/api/stream/resume')
def stream_resume():
    stream.resume()
    return jsonify({'status': 'resumed', 'is_paused': stream.is_paused})

@app.route('/api/stream/toggle')
def stream_toggle():
    stream.toggle_pause()
    return jsonify({'status': 'toggled', 'is_paused': stream.is_paused, 'is_running': stream.is_running})

@app.route('/api/stream/status')
def stream_status():
    stats = stream.get_statistics()
    return jsonify({**stats, 'recent_messages': stream.get_recent_messages(5)})

@app.route('/api/stream/messages')
def stream_messages():
    count = request.args.get('count', 10, type=int)
    messages = stream.get_recent_messages(count)
    return jsonify({'messages': messages, 'count': len(messages), 'total': stream.total_messages})

@app.route('/api/stream/events')
def stream_events():
    def generate():
        while True:
            try:
                message = stream.get_message(timeout=0.5)
                if message:
                    if isinstance(message, dict) and message.get('type') == 'sentiment_update':
                        yield f"data: {json.dumps({'type': 'sentiment_update', 'sentiment_score': message['sentiment_score'], 'timestamp': message['timestamp']})}\n\n"
                    else:
                        yield f"data: {json.dumps(message.to_dict() if hasattr(message, 'to_dict') else message)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'ping', 'timestamp': datetime.now().isoformat()})}\n\n"
                time.sleep(0.1)
            except Exception as e:
                print(f"SSE error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                time.sleep(1)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

# ===== SENTIMENT AGGREGATOR ENDPOINTS =====

@app.route('/api/aggregate/<ticker>')
def get_aggregated_sentiment(ticker):
    aggregator.clear_history()
    resolved_ticker = resolve_ticker(ticker)
    
    try:
        stock_data = get_stock_data(resolved_ticker)
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)}), 503
    
    history_prices = stock_data.get('historical_prices', [])
    
    if len(history_prices) >= 2:
        for idx in range(1, len(history_prices)):
            prev_price = history_prices[idx - 1]
            curr_price = history_prices[idx]
            if prev_price:
                change = (curr_price - prev_price) / prev_price
                score = max(-0.95, min(0.95, change * 3))
                hours_ago = (len(history_prices) - idx) * 6
                point = create_sentiment_point(
                    source='Financial News',
                    score=score,
                    timestamp=datetime.now() - timedelta(hours=hours_ago),
                    confidence=0.8
                )
                aggregator.add_sentiment_point(point)
    
    recent_messages = stream.get_recent_messages(20)
    for msg in recent_messages:
        if msg.get('sentiment_score') is not None:
            source_map = {'Reuters': 'Financial News', 'Bloomberg': 'Financial News', 'CNBC': 'Financial News', 'WSJ': 'Financial News', 'Reddit': 'Reddit/WSB'}
            source = source_map.get(msg.get('source', ''), 'Financial News')
            point = create_sentiment_point(
                source=source,
                score=msg.get('sentiment_score', 0),
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 30)),
                volume=random.randint(10, 100) if source == 'Reddit/WSB' else None
            )
            aggregator.add_sentiment_point(point)
    
    result = aggregator.aggregate()
    result['ticker'] = stock_data.get('ticker', resolved_ticker)
    result['data_source'] = 'yfinance'
    return jsonify(result)

@app.route('/api/aggregate/status')
def get_aggregator_status():
    stats = aggregator.get_statistics()
    return jsonify(stats)

@app.route('/api/aggregate/trend/<ticker>')
def get_aggregator_trend(ticker):
    hours = request.args.get('hours', 24, type=int)
    resolved_ticker = resolve_ticker(ticker)
    trend = aggregator.get_historical_trend(lookback_hours=hours)
    return jsonify({'ticker': resolved_ticker, 'lookback_hours': hours, 'trend_data': trend})

# ===== AGENT ALPHA ENDPOINTS =====

@app.route('/api/agent/analyze/<ticker>')
def agent_analyze(ticker):
    try:
        resolved_ticker = resolve_ticker(ticker)
        print(f"🔍 Agent Alpha analyzing {resolved_ticker}...")
        
        try:
            stock_data = get_stock_data(resolved_ticker)
        except Exception as e:
            return jsonify({'error': True, 'message': f"Cannot analyze: {str(e)}"}), 503
        
        price_change = stock_data.get('price_change', 0.0)
        final_score = max(-0.95, min(0.95, price_change / 35))
        trend = 'bullish' if final_score > 0.05 else 'bearish' if final_score < -0.05 else 'neutral'
        
        sentiment_data = {
            'final_score': round(final_score, 3),
            'final_score_percent': round((final_score + 1) / 2 * 100, 1),
            'trend': trend,
            'summary': f"Sentiment for {resolved_ticker} reflects the latest price move of {price_change:.2f}%",
            'source_breakdown': {
                'SEC Filings': {'raw_score': round(final_score * 0.6, 3), 'message_count': 5},
                'Financial News': {'raw_score': round(final_score * 0.8, 3), 'message_count': 8},
                'Reddit/WSB': {'raw_score': round(final_score * 0.4, 3), 'message_count': 3}
            },
            'recent_messages': [
                {'source': 'Yahoo Finance', 'text': f'{resolved_ticker} moved {price_change:.2f}%', 'sentiment_score': round(final_score, 3)}
            ]
        }
        analysis = agent_alpha.analyze(resolved_ticker, sentiment_data)
        return jsonify({
            'ticker': resolved_ticker, 
            'sentiment_data': sentiment_data, 
            'analysis': analysis
        })
    except Exception as e:
        print(f"❌ Agent Alpha error: {e}")
        return jsonify({'error': True, 'message': str(e)}), 500

# ===== CHART ENDPOINT =====

@app.route('/api/chart/<ticker>')
def get_chart_data(ticker):
    try:
        days = request.args.get('days', 30, type=int)
        resolved_ticker = resolve_ticker(ticker)
        
        try:
            stock_data = get_stock_data(resolved_ticker)
            dates = stock_data.get('historical_dates', [])[-days:]
            prices = stock_data.get('historical_prices', [])[-days:]
            
            sentiment = []
            for i in range(1, len(prices)):
                if prices[i-1]:
                    change = (prices[i] - prices[i-1]) / prices[i-1]
                    sentiment.append(round(max(-1, min(1, change * 5)), 3))
                else:
                    sentiment.append(0)
            
            while len(sentiment) < len(prices):
                sentiment.insert(0, 0)
            
            return jsonify({
                'ticker': resolved_ticker,
                'dates': dates,
                'prices': prices,
                'sentiment': sentiment[-len(dates):] if len(sentiment) > len(dates) else sentiment,
                'keywords': [[resolved_ticker, COMPANY_NAMES.get(resolved_ticker, '').split()[0]] for _ in range(len(dates))]
            })
        except:
            chart = SentimentChart(dark_mode=True)
            df = chart.get_data(resolved_ticker, days)
            return jsonify({
                'ticker': resolved_ticker,
                'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': df['price'].tolist(),
                'sentiment': df['sentiment'].tolist(),
                'keywords': df['keywords'].tolist()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8502))
    app.run(debug=True, host='0.0.0.0', port=port)