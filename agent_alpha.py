"""
Agent Alpha: AI Financial Explainer & Counter-Argument Generator
Uses LLM to provide cynical, quantitative financial analysis
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import requests


class AgentAlpha:
    """
    AI Financial Agent that generates bull/bear analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent with API key
        
        Args:
            api_key: OpenAI API key (or from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = "gpt-3.5-turbo"  # or "gpt-4" for better quality
        
    def generate_prompt(self, ticker: str, sentiment_data: Dict) -> str:
        """
        Generate a comprehensive prompt for the LLM
        
        Args:
            ticker: Stock ticker symbol
            sentiment_data: Dictionary containing sentiment information
        
        Returns:
            Formatted prompt string
        """
        # Extract data
        final_score = sentiment_data.get('final_score', 0)
        score_percent = sentiment_data.get('final_score_percent', 50)
        trend = sentiment_data.get('trend', 'neutral')
        summary = sentiment_data.get('summary', 'No summary available')
        
        # Get source breakdown
        source_breakdown = sentiment_data.get('source_breakdown', {})
        source_text = ""
        for source, data in source_breakdown.items():
            if data.get('message_count', 0) > 0:
                source_text += f"- {source}: Score {data.get('raw_score', 0):.2f} ({data.get('message_count')} messages)\n"
        
        # Get recent messages for context
        recent_messages = sentiment_data.get('recent_messages', [])
        messages_text = ""
        for msg in recent_messages[-5:]:  # Last 5 messages
            messages_text += f"[{msg.get('source', 'Unknown')}] {msg.get('text', '')} (Score: {msg.get('sentiment_score', 0):.2f})\n"
        
        # Build the prompt
        prompt = f"""
You are "Agent Alpha", a cynical, quantitative financial analyst with a contrarian edge. You are analyzing {ticker} stock.

## Current Market Data:
- Overall Sentiment Score: {score_percent}% ({'Bullish' if final_score > 0.2 else 'Bearish' if final_score < -0.2 else 'Neutral'})
- Trend: {trend.upper()}
- Summary: {summary}

## Source Breakdown:
{source_text}

## Recent Sentiment Messages:
{messages_text}

## Your Task:
Provide a structured analysis with TWO distinct sections:

### The Bull Case Strategy
- Synthesize why the positive sentiment indicators matter
- What fundamentals support the optimism
- What specific catalysts could drive the price up
- Be concise but insightful (2-3 paragraphs)

### The Bear Counter-Argument
- Be HIGHLY CRITICAL and adversarial
- Challenge the positive noise aggressively
- Highlight hidden risks in the negative sentiment data
- What could cause a catastrophic downturn
- Be cynical and contrarian (2-3 paragraphs)

### Final Verdict
- Your personal recommendation (BUY/HOLD/SELL)
- Risk level (LOW/MEDIUM/HIGH)
- 1 sentence summary of your thesis

**Rules:**
- Be quantitative and data-driven
- Sound like a professional analyst
- Be slightly cynical and challenging
- No disclaimers or "not financial advice" nonsense
- Just give straight, honest analysis
"""
        return prompt
    
    def call_api(self, prompt: str) -> str:
        """
        Call the LLM API
        
        Args:
            prompt: The formatted prompt
        
        Returns:
            Generated analysis text
        """
        # Using OpenAI API
        if self.api_key:
            try:
                import openai
                openai.api_key = self.api_key
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a cynical, quantitative financial analyst named Agent Alpha. Provide honest, data-driven analysis without disclaimers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except ImportError:
                print("OpenAI package not installed. Using mock response.")
                return self._generate_mock_analysis(prompt)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._generate_mock_analysis(prompt)
        
        # Fallback to mock analysis
        return self._generate_mock_analysis(prompt)
    
    def _generate_mock_analysis(self, prompt: str) -> str:
        """
        Generate mock analysis for testing (no API key)
        """
        # Extract ticker from prompt
        ticker = "the stock"
        lines = prompt.split('\n')
        for line in lines:
            if 'analyzing' in line.lower():
                parts = line.split('analyzing')
                if len(parts) > 1:
                    ticker = parts[1].strip()
                break
        
        return f"""
## The Bull Case Strategy

The sentiment data for {ticker} shows a cautiously optimistic picture. Institutional sources are showing some positive signals, suggesting smart money is accumulating positions. The recent price action indicates momentum is building.

Key catalysts include potential earnings beats, market share expansion, and the current macroeconomic environment. The sentiment trend, while neutral, could be interpreted as stability in an uncertain market.

## The Bear Counter-Argument

This is where I get cynical. The neutral sentiment is actually BEARISH in disguise. Institutional sentiment is negative, which retail traders are completely ignoring. The recent price surge looks suspiciously like a dead cat bounce.

Watch out for:
- An earnings miss that shatters this fragile sentiment
- A sudden rise in bond yields killing growth stock valuations
- The sentiment being artificially inflated by a small group of vocal retail traders

## Final Verdict

**Recommendation: HOLD** (with tight stop-loss at 5% below current price)

**Risk Level: HIGH** - This is a momentum play with weak fundamental backing

**Thesis:** Retail euphoria is propping up this price, but institutional money is quietly exiting. This is a classic distribution pattern before a significant correction.
"""
    
    def analyze(self, ticker: str, sentiment_data: Dict) -> str:
        """
        Main analysis function - generates the agent's response
        
        Args:
            ticker: Stock ticker
            sentiment_data: Sentiment data from aggregator
        
        Returns:
            Formatted markdown analysis
        """
        prompt = self.generate_prompt(ticker, sentiment_data)
        analysis = self.call_api(prompt)
        return analysis