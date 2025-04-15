"""
Main analyzer class for orchestrating Twitter sentiment analysis pipeline.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

from twitter_sentiment.core.data_fetcher import TwitterDataFetcher
from twitter_sentiment.core.sentiment_engine import SentimentEngine
from twitter_sentiment.visualization.word_clouds import generate_sentiment_wordclouds
from twitter_sentiment.visualization.charts import (
    create_sentiment_bar_chart,
    create_hashtag_bar_chart,
    create_sentiment_by_hashtag_chart
)
from twitter_sentiment.visualization.dashboard import (
    create_dashboard,
    create_sentiment_breakdown_dashboard
)
from twitter_sentiment.utils.text_processing import extract_hashtags
from twitter_sentiment.utils.io import save_to_json, save_to_csv
from twitter_sentiment.config import DEFAULT_OUTPUT_PATHS

# Set up logging
logger = logging.getLogger(__name__)


class TwitterSentimentAnalyzer:
    """
    Main class for orchestrating the Twitter sentiment analysis pipeline.
    """
    
    def __init__(
        self, 
        bearer_token: Optional[str] = None,
        use_spacy: bool = True,
        n_workers: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the Twitter sentiment analyzer.
        
        Args:
            bearer_token: Twitter API bearer token
            use_spacy: Whether to use spaCy for NLP features
            n_workers: Number of worker threads for parallel processing
            output_dir: Directory for output files
        """
        self.data_fetcher = TwitterDataFetcher(bearer_token)
        self.sentiment_engine = SentimentEngine(use_spacy, n_workers)
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(DEFAULT_OUTPUT_PATHS["visualization"]).parent
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate full output paths
        self.output_paths = {
            key: str(self.output_dir / Path(path).name)
            for key, path in DEFAULT_OUTPUT_PATHS.items()
        }
        
        logger.info(f"Initialized TwitterSentimentAnalyzer with output directory: {self.output_dir}")
    
    def analyze(
        self,
        query: str,
        max_results: int = 100,
        save_results: bool = True,
        create_visualizations: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            query: Search query for Twitter API
            max_results: Maximum number of results to fetch
            save_results: Whether to save results to files
            create_visualizations: Whether to create visualizations
            
        Returns:
            Tuple of (analyzed_tweets, summary)
        """
        logger.info(f"Starting sentiment analysis for query: '{query}'")
        
        # 1. Fetch tweets
        tweets = self.data_fetcher.fetch_tweets(query, max_results)
        
        # 2. Analyze sentiments
        analyzed_tweets = self.sentiment_engine.analyze_tweets(tweets)
        
        # 3. Get sentiment distribution
        summary = self.sentiment_engine.get_sentiment_distribution(analyzed_tweets)
        
        # Log summary
        logger.info("Sentiment Analysis Summary:")
        logger.info(f"  Positive: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)")
        logger.info(f"  Negative: {summary['negative_count']} ({summary['negative_percentage']:.1f}%)")
        logger.info(f"  Total analyzed: {summary['total_count']}")
        
        # 4. Save results if requested
        if save_results:
            self._save_results(analyzed_tweets)
        
        # 5. Create visualizations if requested
        if create_visualizations:
            self._create_visualizations(analyzed_tweets, summary)
        
        return analyzed_tweets, summary
    
    def analyze_by_hashtag(
        self,
        query: str,
        max_results: int = 100,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment broken down by hashtag.
        
        Args:
            query: Search query for Twitter API
            max_results: Maximum number of results to fetch
            save_results: Whether to save results to files
            
        Returns:
            List of dictionaries with hashtag sentiment information
        """
        logger.info(f"Starting hashtag sentiment analysis for query: '{query}'")
        
        # 1. Run normal analysis
        analyzed_tweets, _ = self.analyze(query, max_results, save_results, False)
        
        # 2. Analyze hashtags
        hashtag_counts = Counter()
        for tweet in analyzed_tweets:
            hashtags = extract_hashtags(tweet["text"])
            hashtag_counts.update(hashtags)
        
        # Get the most common hashtags
        top_hashtags = hashtag_counts.most_common(10)
        
        logger.info("\nTop Hashtags:")
        for i, (hashtag, count) in enumerate(top_hashtags, 1):
            logger.info(f"  {i}. #{hashtag} ({count})")
        
        # 3. Analyze sentiment by hashtag
        hashtag_sentiment = []
        
        if top_hashtags:
            logger.info("\nSentiment by Popular Hashtags:")
            
            for hashtag, count in top_hashtags:
                hashtag_pattern = f"#{hashtag}"
                
                # Filter tweets containing this hashtag
                hashtag_tweets = [
                    tweet for tweet in analyzed_tweets
                    if hashtag_pattern.lower() in tweet["text"].lower()
                ]
                
                if hashtag_tweets:
                    positive_count = sum(1 for tweet in hashtag_tweets 
                                        if tweet["sentiment"] == "positive")
                    positive_percentage = (positive_count / len(hashtag_tweets)) * 100
                    
                    logger.info(f"  #{hashtag}: {positive_percentage:.1f}% positive ({len(hashtag_tweets)} tweets)")
                    
                    hashtag_sentiment.append({
                        "hashtag": hashtag,
                        "count": count,
                        "tweet_count": len(hashtag_tweets),
                        "positive_count": positive_count,
                        "positive_percentage": positive_percentage
                    })
        
        # 4. Create visualization
        if hashtag_sentiment:
            create_sentiment_by_hashtag_chart(
                hashtag_sentiment,
                self.output_paths["hashtag_sentiment"],
                title=f"Sentiment by Hashtag for '{query}'"
            )
            logger.info(f"Hashtag sentiment chart saved to {self.output_paths['hashtag_sentiment']}")
        
        # 5. Save results if requested
        if save_results and hashtag_sentiment:
            hashtag_file = Path(self.output_dir) / "hashtag_sentiment.json"
            save_to_json(hashtag_sentiment, hashtag_file)
            logger.info(f"Hashtag sentiment data saved to {hashtag_file}")
        
        return hashtag_sentiment
    
    def _save_results(self, analyzed_tweets: List[Dict[str, Any]]) -> None:
        """Save analysis results to files."""
        # Save to CSV
        save_to_csv(analyzed_tweets, self.output_paths["csv"])
        logger.info(f"Results saved to CSV: {self.output_paths['csv']}")
        
        # Save to JSON
        save_to_json(analyzed_tweets, self.output_paths["json"])
        logger.info(f"Results saved to JSON: {self.output_paths['json']}")
    
    def _create_visualizations(
        self, 
        analyzed_tweets: List[Dict[str, Any]], 
        summary: Dict[str, Any]
    ) -> None:
        """Create visualizations from analysis results."""
        # 1. Create word clouds for positive and negative tweets
        generate_sentiment_wordclouds(
            analyzed_tweets,
            self.output_paths["wordcloud_positive"],
            self.output_paths["wordcloud_negative"]
        )
        logger.info(f"Word clouds saved to {self.output_paths['wordcloud_positive']} and "
                   f"{self.output_paths['wordcloud_negative']}")
        
        # 2. Create sentiment bar chart
        create_sentiment_bar_chart(
            summary,
            self.output_paths["visualization"],
            title="Twitter Sentiment Analysis"
        )
        logger.info(f"Sentiment bar chart saved to {self.output_paths['visualization']}")
        
        # 3. Create dashboard visualization
        create_dashboard(
            analyzed_tweets,
            summary,
            Path(self.output_dir) / "dashboard.png",
            title="Twitter Sentiment Analysis Dashboard"
        )
        logger.info(f"Dashboard saved to {self.output_dir / 'dashboard.png'}")
        
        # 4. Create sentiment breakdown dashboard
        create_sentiment_breakdown_dashboard(
            analyzed_tweets,
            Path(self.output_dir) / "sentiment_breakdown.png",
            title="Sentiment Analysis Breakdown"
        )
        logger.info(f"Sentiment breakdown dashboard saved to {self.output_dir / 'sentiment_breakdown.png'}")
        
        # 5. Create hashtag visualization
        hashtag_counts = Counter()
        for tweet in analyzed_tweets:
            hashtags = extract_hashtags(tweet["text"])
            hashtag_counts.update(hashtags)
        
        top_hashtags = hashtag_counts.most_common(10)
        
        if top_hashtags:
            create_hashtag_bar_chart(
                top_hashtags,
                Path(self.output_dir) / "hashtags.png",
                title="Top Hashtags"
            )
            logger.info(f"Hashtag chart saved to {self.output_dir / 'hashtags.png'}")