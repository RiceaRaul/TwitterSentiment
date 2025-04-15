__version__ = '0.1.0'

# Import config to make it available to all modules

from twitter_sentiment.core.analyzer import TwitterSentimentAnalyzer
from twitter_sentiment.core.data_fetcher import TwitterDataFetcher
from twitter_sentiment.core.sentiment_engine import SentimentEngine

# Expose main classes at the package level for easier imports
__all__ = [
    'TwitterSentimentAnalyzer',  # Main user-facing class
    'TwitterDataFetcher',        # For advanced usage
    'SentimentEngine',           # For advanced usage
    'config',                    # Configuration module
]