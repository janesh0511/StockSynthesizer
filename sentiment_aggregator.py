"""
SentimentAggregator: Advanced Sentiment Aggregation Engine
Combines multiple sources with weighting, time decay, and volume normalization
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class SentimentPoint:
    """Individual sentiment data point from a source"""
    source: str
    score: float  # -1.0 to +1.0
    timestamp: datetime
    volume: Optional[int] = None  # For Reddit/WSB volume tracking
    text: Optional[str] = None
    confidence: Optional[float] = 1.0


@dataclass
class SourceConfig:
    """Configuration for each data source"""
    weight: float
    credibility_score: float  # 0-1 scale
    max_age_hours: float = 168  # 7 days default
    volume_threshold: Optional[int] = None
    volume_cap: Optional[int] = None


class SentimentAggregator:
    """
    Advanced Sentiment Aggregation Engine with:
    - Source weighting with credibility scores
    - Time decay (exponential decay)
    - Volume normalization for Reddit/WSB
    - Dynamic summary generation
    """
    
    # Default configuration for all sources
    DEFAULT_CONFIG = {
        'SEC Filings': SourceConfig(
            weight=0.40,
            credibility_score=0.95,
            max_age_hours=336  # 14 days
        ),
        'Earnings Transcripts': SourceConfig(
            weight=0.20,
            credibility_score=0.85,
            max_age_hours=168  # 7 days
        ),
        'Financial News': SourceConfig(
            weight=0.30,
            credibility_score=0.70,
            max_age_hours=72  # 3 days
        ),
        'Reddit/WSB': SourceConfig(
            weight=0.10,
            credibility_score=0.30,
            max_age_hours=24,  # 1 day
            volume_threshold=50,
            volume_cap=200
        )
    }
    
    def __init__(self, custom_config: Optional[Dict[str, SourceConfig]] = None):
        """
        Initialize the aggregator with optional custom configuration
        
        Args:
            custom_config: Dictionary of source configurations to override defaults
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            self.config.update(custom_config)
        
        # Store recent sentiment history for trend analysis
        self.sentiment_history: List[SentimentPoint] = []
        self.max_history = 1000
        
        # Track volume spikes for normalization
        self.volume_history: Dict[str, List[int]] = {
            source: [] for source in self.config.keys()
        }
        self.max_volume_history = 100
        
    def add_sentiment_point(self, point: SentimentPoint) -> None:
        """
        Add a new sentiment data point to the history
        
        Args:
            point: SentimentPoint object with source, score, timestamp
        """
        self.sentiment_history.append(point)
        
        # Track volume for normalization
        if point.volume is not None and point.source in self.volume_history:
            self.volume_history[point.source].append(point.volume)
            if len(self.volume_history[point.source]) > self.max_volume_history:
                self.volume_history[point.source] = self.volume_history[point.source][-self.max_volume_history:]
        
        # Limit history size
        if len(self.sentiment_history) > self.max_history:
            self.sentiment_history = self.sentiment_history[-self.max_history:]
    
    def add_batch_sentiment(self, points: List[SentimentPoint]) -> None:
        """Add multiple sentiment points at once"""
        for point in points:
            self.add_sentiment_point(point)
    
    def _calculate_time_decay(self, timestamp: datetime, max_age_hours: float) -> float:
        """
        Calculate exponential time decay factor
        
        Args:
            timestamp: The timestamp of the sentiment point
            max_age_hours: Maximum age in hours before score decays to 0
        
        Returns:
            decay_factor: 0.0 to 1.0, where 1.0 is most recent
        """
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        
        if age_hours <= 0:
            return 1.0
        if age_hours >= max_age_hours:
            return 0.0
        
        # Exponential decay: e^(-λ * age)
        # λ (decay constant) = -ln(0.01) / max_age_hours
        # This gives ~99% decay at max_age_hours
        decay_constant = -math.log(0.01) / max_age_hours
        decay_factor = math.exp(-decay_constant * age_hours)
        
        return max(0.0, min(1.0, decay_factor))
    
    def _normalize_volume(self, source: str, raw_volume: int) -> float:
        """
        Normalize volume using sigmoid function with dynamic threshold
        
        Args:
            source: The source name
            raw_volume: Raw volume count
        
        Returns:
            normalized_volume: 0.0 to 1.0
        """
        config = self.config.get(source)
        if not config or not config.volume_threshold:
            return 1.0
        
        # Calculate rolling average volume
        if source in self.volume_history and self.volume_history[source]:
            avg_volume = np.mean(self.volume_history[source][-20:])  # Last 20 points
            # Use the higher of rolling average or threshold
            threshold = max(config.volume_threshold, avg_volume * 0.5)
        else:
            threshold = config.volume_threshold
        
        # Sigmoid normalization: 1 / (1 + e^(-k*(x - threshold)))
        # This saturates at high volumes to prevent extreme spikes
        k = 0.02  # Steepness of the sigmoid
        normalized = 1 / (1 + math.exp(-k * (raw_volume - threshold)))
        
        # Cap at 1.0 to prevent over-normalization
        return min(1.0, normalized)
    
    def _calculate_source_metrics(self, source: str) -> Dict:
        """
        Calculate metrics for a specific source
        
        Returns:
            Dictionary with source metrics
        """
        points = [p for p in self.sentiment_history if p.source == source]
        
        if not points:
            return {
                'score': 0.0,
                'weighted_score': 0.0,
                'count': 0,
                'volume': 0,
                'decay_factor': 0.0,
                'normalized_volume': 0.0,
                'credibility_score': 0.0,
                'final_weight': 0.0,
                'effective_score': 0.0
            }
        
        config = self.config.get(source)
        if not config:
            return {'score': 0.0, 'count': 0}
        
        # Calculate scores with time decay
        total_weighted_score = 0.0
        total_decay = 0.0
        total_volume = 0
        
        for point in points:
            decay_factor = self._calculate_time_decay(point.timestamp, config.max_age_hours)
            total_decay += decay_factor
            total_weighted_score += point.score * decay_factor
            if point.volume:
                total_volume += point.volume
        
        # Average score weighted by time decay
        avg_score = total_weighted_score / total_decay if total_decay > 0 else 0.0
        avg_decay = total_decay / len(points) if len(points) > 0 else 0.0
        avg_volume = total_volume / len(points) if len(points) > 0 else 0
        
        # Normalize volume
        normalized_volume = self._normalize_volume(source, int(avg_volume)) if avg_volume > 0 else 1.0
        
        # Calculate final weight for this source
        # Weight = credibility_score * decay_factor * normalized_volume
        final_weight = config.credibility_score * avg_decay * normalized_volume
        
        return {
            'score': avg_score,
            'weighted_score': total_weighted_score,
            'count': len(points),
            'volume': total_volume,
            'avg_volume': avg_volume,
            'decay_factor': avg_decay,
            'normalized_volume': normalized_volume,
            'credibility_score': config.credibility_score,
            'final_weight': final_weight,
            'effective_score': avg_score * final_weight,
            'base_weight': config.weight
        }
    
    def _calculate_trend_shift(self) -> Tuple[str, float]:
        """
        Calculate the trend and magnitude of shift in sentiment
        
        Returns:
            Tuple of (direction, magnitude)
        """
        if len(self.sentiment_history) < 5:
            return ('neutral', 0.0)
        
        # Sort by timestamp
        sorted_points = sorted(self.sentiment_history, key=lambda x: x.timestamp)
        
        # Take last 10 points for short-term trend
        recent_points = sorted_points[-10:] if len(sorted_points) >= 10 else sorted_points
        
        # Split into old and new
        split_idx = max(1, len(recent_points) // 2)
        old_scores = [p.score for p in recent_points[:split_idx]]
        new_scores = [p.score for p in recent_points[split_idx:]]
        
        old_avg = np.mean(old_scores) if old_scores else 0
        new_avg = np.mean(new_scores) if new_scores else 0
        
        shift = new_avg - old_avg
        
        if shift > 0.1:
            return ('bullish', shift)
        elif shift < -0.1:
            return ('bearish', shift)
        else:
            return ('neutral', shift)
    
    def _identify_top_sources(self) -> List[Tuple[str, float]]:
        """
        Identify sources with highest contribution to the final score
        
        Returns:
            List of (source, contribution) tuples sorted by contribution
        """
        contributions = []
        for source in self.config.keys():
            metrics = self._calculate_source_metrics(source)
            if metrics['count'] > 0:
                # Calculate contribution to final score
                contribution = metrics['effective_score'] * metrics['base_weight']
                contributions.append((source, contribution))
        
        return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    
    def _generate_summary(self, final_score: float, source_scores: Dict, trend: Tuple[str, float]) -> str:
        """
        Generate a dynamic 2-sentence summary explaining the sentiment
        
        Args:
            final_score: The aggregated score
            source_scores: Dictionary of source scores
            trend: Tuple of (direction, magnitude)
        
        Returns:
            A 2-sentence summary
        """
        direction = "bullish" if final_score > 0.2 else "bearish" if final_score < -0.2 else "neutral"
        strength = "strongly" if abs(final_score) > 0.6 else "moderately" if abs(final_score) > 0.3 else "slightly"
        
        # Identify top sources
        top_sources = self._identify_top_sources()
        
        # Build first sentence
        if top_sources:
            top_source, contribution = top_sources[0]
            sentiment_label = "positive" if contribution > 0 else "negative"
            
            # Determine driving factor
            if top_source in ['SEC Filings', 'Earnings Transcripts']:
                driver = f"institutional sentiment from {top_source}"
            elif top_source in ['Reddit/WSB']:
                driver = f"retail sentiment from {top_source}"
            else:
                driver = f"sentiment from {top_source}"
            
            sentence1 = f"Overall sentiment is {strength} {direction} ({final_score:.2f}), driven primarily by {sentiment_label} {driver}."
        else:
            sentence1 = f"Overall sentiment is {strength} {direction} ({final_score:.2f}) with balanced contributions from all sources."
        
        # Build second sentence
        trend_direction, trend_magnitude = trend
        if trend_direction == 'bullish' and trend_magnitude > 0.1:
            sentence2 = f"Sentiment is strengthening ({trend_magnitude:.2f} shift) as positive signals outweigh negative sentiment from recent data."
        elif trend_direction == 'bearish' and trend_magnitude > 0.1:
            sentence2 = f"Sentiment is weakening ({trend_magnitude:.2f} shift) as negative signals begin to dominate the aggregation."
        else:
            # Mention specific source dynamics
            if len(top_sources) > 1:
                second_source, second_contrib = top_sources[1]
                sentence2 = f"The sentiment is being moderated by {second_source} sentiment ({second_contrib:.2f}) which offsets the primary signal."
            else:
                sentence2 = "Sentiment is relatively stable with no significant shifts detected in the recent data."
        
        return f"{sentence1} {sentence2}"
    
    def aggregate(self) -> Dict:
        """
        Calculate the final aggregated sentiment score
        
        Returns:
            Dictionary with comprehensive aggregation results
        """
        if not self.sentiment_history:
            return {
                'final_score': 0.0,
                'source_scores': {},
                'summary': "No sentiment data available for aggregation.",
                'trend': 'neutral',
                'timestamp': datetime.now().isoformat(),
                'data_points': 0,
                'source_breakdown': {}
            }
        
        # Calculate metrics for each source
        source_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source in self.config.keys():
            metrics = self._calculate_source_metrics(source)
            source_scores[source] = metrics
            
            if metrics['count'] > 0:
                # Apply source weight
                source_weight = self.config[source].weight
                effective_score = metrics['score'] * metrics['decay_factor'] * metrics['normalized_volume']
                weighted_sum += effective_score * source_weight
                total_weight += source_weight * metrics['decay_factor'] * metrics['normalized_volume']
        
        # Calculate final score
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        final_score = max(-1.0, min(1.0, final_score))  # Clamp to [-1, 1]
        
        # Calculate trend
        trend = self._calculate_trend_shift()
        
        # Generate summary
        summary = self._generate_summary(final_score, source_scores, trend)
        
        # Prepare source breakdown for response
        source_breakdown = {}
        for source, metrics in source_scores.items():
            if metrics['count'] > 0:
                source_breakdown[source] = {
                    'raw_score': round(metrics['score'], 3),
                    'weighted_score': round(metrics['effective_score'], 3),
                    'decay_factor': round(metrics['decay_factor'], 3),
                    'volume_normalized': round(metrics['normalized_volume'], 3),
                    'credibility_score': metrics['credibility_score'],
                    'message_count': metrics['count'],
                    'final_weight': round(metrics['final_weight'], 3),
                    'volume': metrics['volume']
                }
        
        # Find most influential source
        influential_sources = []
        for source, metrics in source_scores.items():
            if metrics['count'] > 0 and abs(metrics['score']) > 0.1:
                influential_sources.append({
                    'source': source,
                    'contribution': round(metrics['score'] * metrics['decay_factor'] * metrics['normalized_volume'], 3),
                    'sentiment': 'positive' if metrics['score'] > 0 else 'negative'
                })
        
        # Sort by contribution
        influential_sources.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'final_score': round(final_score, 3),
            'final_score_percent': round((final_score + 1) / 2 * 100, 1),
            'source_scores': source_scores,
            'source_breakdown': source_breakdown,
            'summary': summary,
            'trend': trend[0],
            'trend_magnitude': round(trend[1], 3),
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.sentiment_history),
            'influential_sources': influential_sources[:3],  # Top 3
            'weighted_sum': round(weighted_sum, 4),
            'total_weight': round(total_weight, 4)
        }
    
    def get_historical_trend(self, lookback_hours: int = 24) -> List[Dict]:
        """
        Get historical sentiment trend data for the specified lookback period
        
        Args:
            lookback_hours: Number of hours to look back
        
        Returns:
            List of sentiment data points with timestamps
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent_points = [p for p in self.sentiment_history if p.timestamp >= cutoff]
        
        # Aggregate by hour for smoother trend
        hourly_data = {}
        for point in recent_points:
            hour_key = point.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {'scores': [], 'count': 0, 'sources': {}}
            hourly_data[hour_key]['scores'].append(point.score)
            hourly_data[hour_key]['count'] += 1
            if point.source not in hourly_data[hour_key]['sources']:
                hourly_data[hour_key]['sources'][point.source] = []
            hourly_data[hour_key]['sources'][point.source].append(point.score)
        
        # Convert to list
        trend_data = []
        for hour, data in sorted(hourly_data.items()):
            avg_score = np.mean(data['scores']) if data['scores'] else 0
            source_avg = {}
            for source, scores in data['sources'].items():
                source_avg[source] = np.mean(scores)
            
            trend_data.append({
                'timestamp': hour.isoformat(),
                'score': round(avg_score, 3),
                'count': data['count'],
                'source_averages': source_avg
            })
        
        return trend_data
    
    def clear_history(self) -> None:
        """Clear all stored sentiment history"""
        self.sentiment_history.clear()
        self.volume_history = {source: [] for source in self.config.keys()}
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the current sentiment data
        
        Returns:
            Dictionary with statistics
        """
        if not self.sentiment_history:
            return {'total_points': 0, 'sources': {}}
        
        stats = {
            'total_points': len(self.sentiment_history),
            'sources': {},
            'time_range': {
                'oldest': min(p.timestamp for p in self.sentiment_history).isoformat(),
                'newest': max(p.timestamp for p in self.sentiment_history).isoformat()
            }
        }
        
        for source in self.config.keys():
            source_points = [p for p in self.sentiment_history if p.source == source]
            if source_points:
                scores = [p.score for p in source_points]
                stats['sources'][source] = {
                    'count': len(source_points),
                    'avg_score': round(np.mean(scores), 3),
                    'std_score': round(np.std(scores), 3),
                    'min_score': round(min(scores), 3),
                    'max_score': round(max(scores), 3)
                }
        
        return stats


# ===== HELPER FUNCTIONS FOR INTEGRATION =====

def create_sentiment_point(
    source: str,
    score: float,
    timestamp: Optional[datetime] = None,
    volume: Optional[int] = None,
    text: Optional[str] = None,
    confidence: Optional[float] = 1.0
) -> SentimentPoint:
    """
    Helper function to create a SentimentPoint
    
    Args:
        source: Source name (must match config keys)
        score: Sentiment score (-1.0 to +1.0)
        timestamp: Datetime of the sentiment
        volume: Optional volume count (for Reddit/WSB)
        text: Optional text content
        confidence: Optional confidence score
    
    Returns:
        SentimentPoint object
    """
    return SentimentPoint(
        source=source,
        score=score,
        timestamp=timestamp or datetime.now(),
        volume=volume,
        text=text,
        confidence=confidence
    )


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example usage of the SentimentAggregator
    
    # 1. Create aggregator with default config
    aggregator = SentimentAggregator()
    
    # 2. Create sample sentiment data points
    sample_data = [
        SentimentPoint(
            source="SEC Filings",
            score=0.85,
            timestamp=datetime.now() - timedelta(hours=2),
            confidence=0.95
        ),
        SentimentPoint(
            source="Earnings Transcripts",
            score=0.60,
            timestamp=datetime.now() - timedelta(hours=5),
            confidence=0.85
        ),
        SentimentPoint(
            source="Financial News",
            score=0.40,
            timestamp=datetime.now() - timedelta(hours=12),
            confidence=0.70,
            text="AAPL stock surges after analyst upgrade"
        ),
        SentimentPoint(
            source="Reddit/WSB",
            score=-0.30,
            timestamp=datetime.now() - timedelta(hours=3),
            volume=150,
            confidence=0.50,
            text="Bearish sentiment on WSB"
        ),
        SentimentPoint(
            source="SEC Filings",
            score=0.90,
            timestamp=datetime.now() - timedelta(hours=6),
            confidence=0.95
        ),
        SentimentPoint(
            source="Financial News",
            score=0.35,
            timestamp=datetime.now() - timedelta(hours=24),
            confidence=0.70
        ),
        SentimentPoint(
            source="Reddit/WSB",
            score=0.50,
            timestamp=datetime.now() - timedelta(hours=1),
            volume=200,
            confidence=0.50,
            text="Bullish sentiment on WSB"
        )
    ]
    
    # 3. Add all data points
    aggregator.add_batch_sentiment(sample_data)
    
    # 4. Perform aggregation
    result = aggregator.aggregate()
    
    # 5. Print results
    print("=" * 70)
    print("SENTIMENT AGGREGATION RESULTS")
    print("=" * 70)
    print(f"📊 Final Score: {result['final_score']:.3f} ({result['final_score_percent']:.1f}%)")
    print(f"📈 Trend: {result['trend']} (Magnitude: {result['trend_magnitude']:.3f})")
    print(f"📝 Data Points: {result['data_points']}")
    print(f"🕐 Timestamp: {result['timestamp']}")
    print("\n" + "─" * 70)
    print("SOURCE BREAKDOWN:")
    print("─" * 70)
    for source, breakdown in result['source_breakdown'].items():
        print(f"\n  📌 {source}:")
        print(f"     Raw Score: {breakdown['raw_score']:.3f}")
        print(f"     Weighted: {breakdown['weighted_score']:.3f}")
        print(f"     Decay Factor: {breakdown['decay_factor']:.3f}")
        print(f"     Volume Normalized: {breakdown['volume_normalized']:.3f}")
        print(f"     Messages: {breakdown['message_count']}")
        print(f"     Volume: {breakdown['volume']}")
    
    print("\n" + "─" * 70)
    print("TOP INFLUENTIAL SOURCES:")
    print("─" * 70)
    for source in result['influential_sources']:
        print(f"  {source['source']}: {source['contribution']:.3f} ({source['sentiment']})")
    
    print("\n" + "─" * 70)
    print("SUMMARY:")
    print("─" * 70)
    print(f"  {result['summary']}")
    print("\n" + "=" * 70)