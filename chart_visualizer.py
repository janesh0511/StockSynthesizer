"""
Interactive Sentiment vs. Price Chart Module
Dual-axis Plotly chart for Flask app
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yfinance as yf


class SentimentChart:
    """Professional dual-axis chart for sentiment vs price visualization"""
    
    def __init__(self, dark_mode=True):
        self.dark_mode = dark_mode
        self.colors = self._get_colors()
        
    def _get_colors(self):
        if self.dark_mode:
            return {
                'background': '#0e1117',
                'grid': '#3a3c44',
                'text': '#fafafa',
                'price_line': '#4a9eff',
                'sentiment_positive': '#2ecc71',
                'sentiment_negative': '#ff6b6b',
                'sentiment_neutral': '#ffd93d',
            }
        else:
            return {
                'background': '#ffffff',
                'grid': '#e0e0e0',
                'text': '#1a1a2e',
                'price_line': '#4a9eff',
                'sentiment_positive': '#27ae60',
                'sentiment_negative': '#e74c3c',
                'sentiment_neutral': '#f39c12',
            }
    
    def get_sentiment_color(self, sentiment: float) -> str:
        if sentiment > 0.3:
            return self.colors['sentiment_positive']
        elif sentiment < -0.3:
            return self.colors['sentiment_negative']
        else:
            return self.colors['sentiment_neutral']
    
    def get_data(self, ticker: str, days: int = 30):
        """Fetch or generate data for the chart"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")
            
            if hist.empty:
                return self._generate_mock_data(ticker, days)
            
            prices = hist['Close'].tolist()
            sentiment_scores = []
            
            keyword_pool = [
                ['earnings', 'beat'], ['revenue', 'growth'], ['market', 'share'],
                ['innovation', 'AI'], ['supply', 'chain'], ['inflation', 'rates'],
                ['analyst', 'upgrade'], ['buyback', 'dividend'], ['lawsuit', 'risk'],
                ['crypto', 'volatility'], ['employment', 'data'], ['consumer', 'spending']
            ]
            
            for i in range(len(prices)):
                if i == 0:
                    sentiment = random.uniform(-0.2, 0.2)
                else:
                    price_change = (prices[i] - prices[i-1]) / prices[i-1]
                    sentiment = price_change * 5 + np.random.randn() * 0.15
                    sentiment = max(-1, min(1, sentiment))
                sentiment_scores.append(round(sentiment, 3))
            
            df = pd.DataFrame({
                'date': hist.index,
                'price': prices,
                'sentiment': sentiment_scores,
                'keywords': [random.choice(keyword_pool) for _ in range(len(prices))]
            })
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._generate_mock_data(ticker, days)
    
    def _generate_mock_data(self, ticker: str, days: int = 30):
        """Generate clean mock data for demonstration"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        base_price = 150 + random.uniform(-20, 20)
        prices = []
        sentiment_scores = []
        
        keyword_pool = [
            ['earnings', 'beat'], ['revenue', 'growth'], ['market', 'share'],
            ['innovation', 'AI'], ['supply', 'chain'], ['inflation', 'rates'],
            ['analyst', 'upgrade'], ['buyback', 'dividend'], ['lawsuit', 'risk'],
            ['crypto', 'volatility'], ['employment', 'data'], ['consumer', 'spending']
        ]
        
        for i in range(len(dates)):
            if i == 0:
                price = base_price
            else:
                change_percent = np.random.randn() * 0.015
                price = prices[-1] * (1 + change_percent)
                price = max(price, base_price * 0.8)
                price = min(price, base_price * 1.2)
            prices.append(round(price, 2))
            
            if i == 0:
                sentiment = random.uniform(-0.2, 0.2)
            else:
                price_change = (prices[-1] - prices[-2]) / prices[-2]
                sentiment = price_change * 5 + np.random.randn() * 0.15
                sentiment = max(-1, min(1, sentiment))
            sentiment_scores.append(round(sentiment, 3))
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'sentiment': sentiment_scores,
            'keywords': [random.choice(keyword_pool) for _ in range(len(dates))]
        })
        return df
    
    def create_chart(self, ticker: str, df: pd.DataFrame):
        """Create the dual-axis Plotly chart"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Price chart (Left Y-Axis)
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['price'],
                mode='lines',
                name='Price',
                line=dict(color='#4a9eff', width=2.5),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Sentiment chart (Right Y-Axis)
        sentiment_colors = [self.get_sentiment_color(s) for s in df['sentiment']]
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['sentiment'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#ffd93d', width=2),
                marker=dict(
                    size=8,
                    color=sentiment_colors,
                    line=dict(width=1, color='#1e1f26')
                ),
                fill='tozeroy',
                fillcolor='rgba(255, 217, 61, 0.15)',
                hovertemplate=(
                    '<b>%{x|%Y-%m-%d}</b><br>'
                    'Sentiment: %{y:.3f}<br>'
                    'Keywords: %{customdata}<extra></extra>'
                ),
                customdata=df['keywords'].apply(lambda x: ', '.join(x))
            ),
            secondary_y=True
        )
        
        # Layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#fafafa'),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(30, 31, 38, 0.95)',
                font_size=12,
                font_color='#fafafa',
                bordercolor='rgba(74, 158, 255, 0.3)'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=60, r=60, t=40, b=40),
            height=500,
            xaxis=dict(
                title=dict(text='Date', font=dict(color='#b0b0b0')),
                gridcolor='#3a3c44',
                tickfont=dict(color='#b0b0b0'),
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(color='#b0b0b0')),
                gridcolor='#3a3c44',
                tickfont=dict(color='#b0b0b0'),
                side='left',
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis2=dict(
                title=dict(text='Sentiment Score', font=dict(color='#b0b0b0')),
                tickfont=dict(color='#b0b0b0'),
                side='right',
                overlaying='y',
                range=[-1.2, 1.2],
                showgrid=False
            )
        )
        
        # Add annotations
        fig.add_annotation(
            x=df['date'].iloc[-1],
            y=0.6,
            text='🟢 Bullish',
            showarrow=False,
            font=dict(size=10, color='#2ecc71'),
            xanchor='right',
            secondary_y=True
        )
        
        fig.add_annotation(
            x=df['date'].iloc[-1],
            y=-0.6,
            text='🔴 Bearish',
            showarrow=False,
            font=dict(size=10, color='#ff6b6b'),
            xanchor='right',
            secondary_y=True
        )
        
        # Add horizontal lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="#2ecc71", opacity=0.4, secondary_y=True)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="#ff6b6b", opacity=0.4, secondary_y=True)
        
        return fig