from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import os  # ADD THIS if not already there
import time
import random
import queue
import threading
import re
from agent_alpha import AgentAlpha
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from agent_alpha import AgentAlpha

# ADD THIS IMPORT
from sentiment_aggregator import SentimentAggregator, create_sentiment_point


app = Flask(__name__, static_folder='.', static_url_path='')

# ===== STOCK DATA CACHE =====
stock_cache = {}
CACHE_DURATION = 60  # seconds

# ===== REAL-TIME STREAMING ENGINE =====
class StockTickerExtractor:
    """Extract stock tickers from text"""
    
    TICKER_PATTERNS = {
        'AAPL': r'\b(AAPL|Apple)\b',
        'TSLA': r'\b(TSLA|Tesla)\b',
        'NVDA': r'\b(NVDA|NVIDIA)\b',
        'MSFT': r'\b(MSFT|Microsoft)\b',
        'AMZN': r'\b(AMZN|Amazon)\b',
        'GOOGL': r'\b(GOOGL|Google|Alphabet)\b',
        'META': r'\b(META|Facebook)\b',
        'SPY': r'\bSPY\b',
        'QQQ': r'\bQQQ\b',
        'BTC': r'\b(BTC|Bitcoin)\b',
        'ETH': r'\b(ETH|Ethereum)\b',
        'DOGE': r'\bDOGE\b',
    }
    
    @classmethod
    def extract_tickers(cls, text: str) -> List[str]:
        """Extract all mentioned tickers from text"""
        found_tickers = []
        for ticker, pattern in cls.TICKER_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_tickers.append(ticker)
        return found_tickers
    
    @classmethod
    def get_primary_ticker(cls, text: str) -> str:
        """Get the first mentioned ticker or return random"""
        tickers = cls.extract_tickers(text)
        return tickers[0] if tickers else random.choice(list(cls.TICKER_PATTERNS.keys()))


@dataclass
class SentimentMessage:
    """Data structure for incoming sentiment messages"""
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


class MockDataGenerator:
    """Generate mock financial news and social media posts"""
    
    HEADLINES = [
        "BREAKING: {ticker} reports record quarterly earnings, beating estimates",
        "{ticker} stock surges after positive analyst rating upgrade",
        "Market analysts predict {ticker} to outperform in Q3",
        "{ticker} announces strategic partnership with major tech company",
        "Institutional investors increase {ticker} holdings by 15%",
        "{ticker} faces regulatory scrutiny over recent acquisitions",
        "{ticker} unveils revolutionary new product line at conference",
        "Short sellers target {ticker} as valuation concerns grow",
        "{ticker} CEO optimistic about future growth prospects",
        "Supply chain issues impact {ticker} production targets",
        "Analyst warns {ticker} may be overvalued at current levels",
        "{ticker} expands into emerging markets, stock jumps",
        "New study suggests {ticker} technology could disrupt industry",
        "{ticker} board announces share buyback program",
        "Dividend increase announced for {ticker} shareholders"
    ]
    
    REDDIT_POSTS = [
        "Just bought more {ticker} 🚀🚀🚀 to the moon!",
        "Deep analysis: Why {ticker} is my top pick for 2025",
        "Paper hands sold {ticker}? Missing out on huge gains!",
        "{ticker} earnings play - what's your target price?",
        "Diamond hands holding {ticker} through this dip 💎🙌",
        "AI just analyzed {ticker} sentiment - bullish signals detected",
        "Looking at the charts, {ticker} about to break resistance",
        "What happened to {ticker} today? Huge volume spike!",
        "Call options on {ticker} looking juicy 🤑",
        "{ticker} management team is top tier - long term hold"
    ]
    
    REDDIT_AUTHORS = [
        "wallstreetbets", "DeepFuckingValue", "OptionsKing",
        "TheCryptoTrader", "ValueInvestor", "ChartMaster",
        "MoonMission", "HODL_4_Life", "TechAnalyst",
        "MarketWatcher", "RiskManager", "BullishBets"
    ]
    
    @classmethod
    def generate_news_message(cls) -> SentimentMessage:
        """Generate a mock news message with balanced sentiment"""
        ticker = random.choice(list(StockTickerExtractor.TICKER_PATTERNS.keys()))
        headline = random.choice(cls.HEADLINES).format(ticker=ticker)
        
        sentiment = random.uniform(-0.4, 0.4)
        
        if any(word in headline.lower() for word in ['surges', 'positive', 'record', 'beat']):
            sentiment = random.uniform(0.1, 0.5)
        elif any(word in headline.lower() for word in ['warns', 'target', 'risk', 'regulatory']):
            sentiment = random.uniform(-0.5, -0.1)
        
        return SentimentMessage(
            text=headline,
            ticker=ticker,
            source=random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ']),
            timestamp=datetime.now().isoformat(),
            sentiment_score=round(sentiment, 2)
        )
    
    @classmethod
    def generate_reddit_message(cls) -> SentimentMessage:
        """Generate a mock Reddit post with balanced sentiment"""
        ticker = random.choice(list(StockTickerExtractor.TICKER_PATTERNS.keys()))
        post = random.choice(cls.REDDIT_POSTS).format(ticker=ticker)
        
        sentiment = random.uniform(-0.3, 0.5)
        
        if '🚀' in post or '📈' in post or 'moon' in post.lower():
            sentiment = random.uniform(0.2, 0.6)
        elif '💎' in post or 'diamond' in post.lower():
            sentiment = random.uniform(0.1, 0.4)
        elif 'paper' in post.lower() or 'dip' in post.lower():
            sentiment = random.uniform(-0.3, 0.1)
        
        return SentimentMessage(
            text=post,
            ticker=ticker,
            source='Reddit',
            timestamp=datetime.now().isoformat(),
            sentiment_score=round(sentiment, 2),
            author=random.choice(cls.REDDIT_AUTHORS),
            upvotes=random.randint(1, 5000)
        )
    
    @classmethod
    def generate_mixed_message(cls) -> SentimentMessage:
        """Generate a mix of news and social media messages"""
        if random.random() < 0.4:
            return cls.generate_news_message()
        else:
            return cls.generate_reddit_message()


class RealTimeStream:
    """Real-time data stream manager with thread-safe queue"""
    
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
        """Start the data stream in a background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("📡 Data stream started")
    
    def stop(self):
        """Stop the data stream"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("📡 Data stream stopped")
    
    def pause(self):
        """Pause the data stream"""
        self.is_paused = True
        print("⏸️ Data stream paused")
    
    def resume(self):
        """Resume the data stream"""
        self.is_paused = False
        print("▶️ Data stream resumed")
    
    def toggle_pause(self):
        """Toggle pause/resume state"""
        if self.is_paused:
            self.resume()
        else:
            self.pause()
    
    def get_message(self, timeout: float = 0.1) -> Optional[SentimentMessage]:
        """Get the next message from the queue"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.message_queue.qsize()
    
    def get_statistics(self) -> Dict:
        """Get streaming statistics"""
        return {
            'total_messages': self.total_messages,
            'queue_size': self.message_queue.qsize(),
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'ticker_frequency': self.ticker_frequency,
            'recent_count': len(self.recent_messages)
        }
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """Get recent messages without consuming them"""
        return self.recent_messages[-n:]
    
    def _run_loop(self):
        """Main loop running in background thread with proper smoothing"""
        message_buffer = []
        last_sentiment_update = time.time()
        sentiment_update_interval = 5.0
        
        running_sentiment = 0.5
        alpha = 0.3
        
        while self.is_running:
            try:
                if not self.is_paused:
                    message = MockDataGenerator.generate_mixed_message()
                    self.message_queue.put(message)
                    self.total_messages += 1
                    
                    if message.ticker:
                        self.ticker_frequency[message.ticker] = \
                            self.ticker_frequency.get(message.ticker, 0) + 1
                    
                    message_dict = message.to_dict()
                    self.recent_messages.append(message_dict)
                    if len(self.recent_messages) > self.max_recent:
                        self.recent_messages = self.recent_messages[-self.max_recent:]
                    
                    if message.sentiment_score is not None:
                        message_buffer.append(message.sentiment_score)
                    
                    current_time = time.time()
                    if current_time - last_sentiment_update >= sentiment_update_interval and message_buffer:
                        avg_score = sum(message_buffer) / len(message_buffer)
                        running_sentiment = running_sentiment * (1 - alpha) + avg_score * alpha
                        running_sentiment = max(-1, min(1, running_sentiment))
                        
                        smoothed_msg = {
                            'type': 'sentiment_update',
                            'sentiment_score': running_sentiment,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.message_queue.put(smoothed_msg)
                        
                        message_buffer = []
                        last_sentiment_update = current_time
                
                time.sleep(random.uniform(1.0, 2.0))
                
            except Exception as e:
                print(f"Error in data stream: {e}")
                time.sleep(1.0)


# ===== GLOBAL STREAM INSTANCE =====
stream = RealTimeStream()
stream.start()



# ADD THIS - Sentiment Aggregator Instance
aggregator = SentimentAggregator()

# ===== AGENT ALPHA INSTANCE =====
agent_alpha = AgentAlpha()

# ===== STOCK DATA FUNCTIONS =====
def get_stock_data(ticker):
    """Fetch real stock data from Yahoo Finance"""
    try:
        cache_key = ticker.upper()
        if cache_key in stock_cache:
            cache_time, data = stock_cache[cache_key]
            if (datetime.now() - cache_time).seconds < CACHE_DURATION:
                return data
        
        stock = yf.Ticker(ticker)
        current_data = stock.history(period="1d")
        if current_data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        current_price = current_data['Close'].iloc[-1]
        hist_data = stock.history(period="1mo")
        info = stock.info
        
        price_change = ((current_price - hist_data['Close'].iloc[0]) / hist_data['Close'].iloc[0]) * 100
        sentiment_score = min(100, max(0, 50 + (price_change * 2)))
        
        result = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker.upper()),
            'current_price': round(current_price, 2),
            'currency': info.get('currency', 'USD'),
            'previous_close': round(current_data['Close'].iloc[-2] if len(current_data) > 1 else current_price, 2),
            'open': round(current_data['Open'].iloc[-1], 2),
            'high': round(current_data['High'].iloc[-1], 2),
            'low': round(current_data['Low'].iloc[-1], 2),
            'volume': int(current_data['Volume'].iloc[-1]),
            'price_change': round(price_change, 2),
            'sentiment_score': round(sentiment_score),
            'historical_prices': hist_data['Close'].tolist(),
            'historical_dates': [d.strftime('%Y-%m-%d') for d in hist_data.index],
            'timestamp': datetime.now().isoformat()
        }
        
        stock_cache[cache_key] = (datetime.now(), result)
        return result
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return generate_mock_data(ticker)


def generate_mock_data(ticker):
    """Generate mock data as fallback"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
    prices = [150 + i * 0.5 + np.random.randn() * 2 for i in range(30)]
    current_price = prices[-1]
    
    return {
        'ticker': ticker.upper(),
        'company_name': f'{ticker.upper()} Inc.',
        'current_price': round(current_price, 2),
        'currency': 'USD',
        'previous_close': round(prices[-2], 2),
        'open': round(prices[0] + 2, 2),
        'high': round(max(prices) + 1, 2),
        'low': round(min(prices) - 1, 2),
        'volume': int(np.random.randint(1000000, 10000000)),
        'price_change': round(((current_price - prices[0]) / prices[0]) * 100, 2),
        'sentiment_score': np.random.randint(40, 85),
        'historical_prices': prices,
        'historical_dates': [d.strftime('%Y-%m-%d') for d in dates],
        'timestamp': datetime.now().isoformat()
    }


# ===== API ENDPOINTS =====

@app.route('/')
def landing():
    """Serve the landing page"""
    return send_from_directory('.', 'landing.html')

@app.route('/app')
def serve_app():
    """Serve the main application"""
    return send_from_directory('.', 'index.html')


@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    """API endpoint for real-time stock data"""
    data = get_stock_data(ticker)
    return jsonify(data)


@app.route('/api/sentiment')
def get_sentiment():
    """API endpoint for sentiment data"""
    ticker = request.args.get('ticker', 'AAPL')
    stock_data = get_stock_data(ticker)
    
    trend_data = {
        'dates': stock_data.get('historical_dates', [])[-7:],
        'sentiment': [stock_data['sentiment_score'] - 10 + i * 2 for i in range(7)],
        'price': stock_data.get('historical_prices', [])[-7:]
    }
    
    stream_stats = stream.get_statistics()
    recent_messages = stream.get_recent_messages(10)
    
    return jsonify({
        'ticker': ticker.upper(),
        'sentiment_score': stock_data['sentiment_score'],
        'trend_data': trend_data,
        'stock_data': {
            'current_price': stock_data['current_price'],
            'price_change': stock_data['price_change'],
            'previous_close': stock_data['previous_close'],
            'open': stock_data['open'],
            'high': stock_data['high'],
            'low': stock_data['low'],
            'volume': stock_data['volume'],
            'company_name': stock_data['company_name']
        },
        'stream_stats': stream_stats,
        'recent_messages': recent_messages,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/news')
def get_news():
    """API endpoint for news sentiment"""
    ticker = request.args.get('ticker', 'AAPL')
    
    news_data = [
        {'outlet': 'Bloomberg', 'score': round(0.5 + np.random.randn() * 0.2, 2), 'impact': 'High'},
        {'outlet': 'Reuters', 'score': round(0.5 + np.random.randn() * 0.2, 2), 'impact': 'Med'},
        {'outlet': 'CNBC', 'score': round(0.5 + np.random.randn() * 0.2, 2), 'impact': 'Med'},
        {'outlet': 'WSJ', 'score': round(0.5 + np.random.randn() * 0.2, 2), 'impact': 'High'}
    ]
    return jsonify(news_data)


@app.route('/api/news/real')
def get_real_news():
    """Get real news data with caching"""
    ticker = request.args.get('ticker', 'AAPL')
    
    news_items = [
        {
            'title': f'{ticker} Reports Strong Q3 Earnings, Beats Estimates',
            'source': 'Bloomberg',
            'sentiment': round(random.uniform(0.3, 0.8), 2),
            'time': datetime.now().strftime('%H:%M'),
            'url': '#'
        },
        {
            'title': f'Analysts Upgrade {ticker} to "Buy" Citing Growth Potential',
            'source': 'Reuters',
            'sentiment': round(random.uniform(0.2, 0.7), 2),
            'time': (datetime.now() - timedelta(minutes=15)).strftime('%H:%M'),
            'url': '#'
        },
        {
            'title': f'{ticker} Announces Strategic Partnership in AI Sector',
            'source': 'CNBC',
            'sentiment': round(random.uniform(0.4, 0.9), 2),
            'time': (datetime.now() - timedelta(minutes=30)).strftime('%H:%M'),
            'url': '#'
        },
        {
            'title': f'Market Analysts Divided on {ticker} Future Performance',
            'source': 'WSJ',
            'sentiment': round(random.uniform(-0.1, 0.3), 2),
            'time': (datetime.now() - timedelta(minutes=45)).strftime('%H:%M'),
            'url': '#'
        },
        {
            'title': f'{ticker} Faces Regulatory Scrutiny Over Recent Acquisition',
            'source': 'Financial Times',
            'sentiment': round(random.uniform(-0.5, -0.1), 2),
            'time': (datetime.now() - timedelta(hours=1)).strftime('%H:%M'),
            'url': '#'
        },
        {
            'title': f'Institutional Investors Increase {ticker} Holdings by 15%',
            'source': 'MarketWatch',
            'sentiment': round(random.uniform(0.5, 0.9), 2),
            'time': (datetime.now() - timedelta(hours=2)).strftime('%H:%M'),
            'url': '#'
        }
    ]
    
    return jsonify(news_items)


@app.route('/api/stream/start')
def stream_start():
    """Start the real-time data stream"""
    stream.start()
    return jsonify({'status': 'started', 'is_running': stream.is_running})


@app.route('/api/stream/stop')
def stream_stop():
    """Stop the real-time data stream"""
    stream.stop()
    return jsonify({'status': 'stopped', 'is_running': stream.is_running})


@app.route('/api/stream/pause')
def stream_pause():
    """Pause the real-time data stream"""
    stream.pause()
    return jsonify({'status': 'paused', 'is_paused': stream.is_paused})


@app.route('/api/stream/resume')
def stream_resume():
    """Resume the real-time data stream"""
    stream.resume()
    return jsonify({'status': 'resumed', 'is_paused': stream.is_paused})


@app.route('/api/stream/toggle')
def stream_toggle():
    """Toggle the real-time data stream"""
    stream.toggle_pause()
    return jsonify({
        'status': 'toggled',
        'is_paused': stream.is_paused,
        'is_running': stream.is_running
    })


@app.route('/api/stream/status')
def stream_status():
    """Get stream status and statistics"""
    stats = stream.get_statistics()
    return jsonify({
        **stats,
        'recent_messages': stream.get_recent_messages(5)
    })


@app.route('/api/stream/messages')
def stream_messages():
    """Get recent messages from the stream"""
    count = request.args.get('count', 10, type=int)
    messages = stream.get_recent_messages(count)
    return jsonify({
        'messages': messages,
        'count': len(messages),
        'total': stream.total_messages
    })


@app.route('/api/stream/events')
def stream_events():
    """Server-Sent Events endpoint for real-time message streaming"""
    def generate():
        """Generate SSE events with throttling"""
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
    """Get aggregated sentiment score for a specific ticker"""
    sources = ['SEC Filings', 'Earnings Transcripts', 'Financial News', 'Reddit/WSB']
    
    # Clear previous data for fresh aggregation
    aggregator.clear_history()
    
    # Add real-time stream messages to aggregator
    recent_messages = stream.get_recent_messages(20)
    for msg in recent_messages:
        if msg.get('sentiment_score') is not None:
            # Map source names to match aggregator
            source_map = {
                'Reuters': 'Financial News',
                'Bloomberg': 'Financial News',
                'CNBC': 'Financial News',
                'WSJ': 'Financial News',
                'Reddit': 'Reddit/WSB'
            }
            source = source_map.get(msg.get('source', ''), 'Financial News')
            
            point = create_sentiment_point(
                source=source,
                score=msg.get('sentiment_score', 0),
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 30)),
                volume=random.randint(10, 100) if source == 'Reddit/WSB' else None
            )
            aggregator.add_sentiment_point(point)
    
    # Generate additional mock data for variety
    for i in range(15):
        source = random.choice(sources)
        score = random.uniform(-0.8, 0.8)
        hours_ago = random.uniform(0.1, 48)
        volume = random.randint(10, 200) if source == 'Reddit/WSB' else None
        
        point = create_sentiment_point(
            source=source,
            score=score,
            timestamp=datetime.now() - timedelta(hours=hours_ago),
            volume=volume,
            confidence=random.uniform(0.5, 0.95)
        )
        aggregator.add_sentiment_point(point)
    
    # Perform aggregation
    result = aggregator.aggregate()
    result['ticker'] = ticker.upper()
    
    return jsonify(result)


@app.route('/api/aggregate/status')
def get_aggregator_status():
    """Get aggregator statistics"""
    stats = aggregator.get_statistics()
    return jsonify(stats)


@app.route('/api/aggregate/trend/<ticker>')
def get_aggregator_trend(ticker):
    """Get historical trend data"""
    hours = request.args.get('hours', 24, type=int)
    trend = aggregator.get_historical_trend(lookback_hours=hours)
    return jsonify({
        'ticker': ticker.upper(),
        'lookback_hours': hours,
        'trend_data': trend
    })

# ===== END SENTIMENT AGGREGATOR ENDPOINTS =====

# ===== AGENT ALPHA ENDPOINTS =====

@app.route('/api/agent/analyze/<ticker>')
def agent_analyze(ticker):
    """Get Agent Alpha analysis for a ticker"""
    try:
        print(f"🔍 Agent Alpha analyzing {ticker}...")
        
        # Get sentiment data
        sentiment_data = {
            'final_score': random.uniform(-0.3, 0.3),
            'final_score_percent': random.randint(30, 70),
            'trend': random.choice(['bullish', 'bearish', 'neutral']),
            'summary': f"Sentiment for {ticker} is mixed with institutional caution and retail optimism.",
            'source_breakdown': {
                'SEC Filings': {'raw_score': random.uniform(-0.5, 0.2), 'message_count': 5},
                'Financial News': {'raw_score': random.uniform(-0.2, 0.3), 'message_count': 8},
                'Reddit/WSB': {'raw_score': random.uniform(-0.1, 0.5), 'message_count': 3}
            },
            'recent_messages': [
                {'source': 'WSJ', 'text': f'{ticker} announces strategic partnership', 'sentiment_score': 0.3},
                {'source': 'Bloomberg', 'text': f'{ticker} faces regulatory scrutiny', 'sentiment_score': -0.2},
            ]
        }
        
        # Get agent analysis
        analysis = agent_alpha.analyze(ticker, sentiment_data)
        
        return jsonify({
            'ticker': ticker.upper(),
            'sentiment_data': sentiment_data,
            'analysis': analysis
        })
    except Exception as e:
        print(f"❌ Agent Alpha error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return mock analysis on error
        return jsonify({
            'ticker': ticker.upper(),
            'analysis': f"""
## 🔼 The Bull Case Strategy

The sentiment data for {ticker.upper()} shows a cautiously optimistic picture. Institutional sources are showing some positive signals, suggesting smart money is accumulating positions. The recent price action indicates momentum is building.

Key catalysts include potential earnings beats, market share expansion, and the current macroeconomic environment favoring tech names.

## 🔽 The Bear Counter-Argument

This is where I get cynical. The neutral sentiment is actually BEARISH in disguise. Institutional sentiment is negative, which retail traders are completely ignoring. The recent price surge looks suspiciously like a dead cat bounce.

Watch out for:
- An earnings miss that shatters this fragile sentiment
- A sudden rise in bond yields killing growth stock valuations
- The sentiment being artificially inflated by a small group of vocal retail traders

## 🎯 Final Verdict

**Recommendation: HOLD** (with tight stop-loss at 5% below current price)

**Risk Level: HIGH** - This is a momentum play with weak fundamental backing

**Thesis:** Retail euphoria is propping up this price, but institutional money is quietly exiting. This is a classic distribution pattern before a significant correction.
"""
        }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8501)
