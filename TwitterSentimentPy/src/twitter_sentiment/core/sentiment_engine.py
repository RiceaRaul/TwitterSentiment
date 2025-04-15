import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

import spacy
from spacy.tokens import Doc
from collections import Counter

from twitter_sentiment.utils.text_processing import (
    preprocess_tweet,
    tokenize_with_regex,
    remove_stopwords
)
from twitter_sentiment.data.lexicons import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    TECH_POSITIVE_WORDS,
    TECH_NEGATIVE_WORDS,
    POSITIVE_EMOTICONS,
    NEGATIVE_EMOTICONS,
    POSITIVE_CENTROIDS,
    NEGATIVE_CENTROIDS
)
from twitter_sentiment.config import (
    SPACY_MODEL,
    SIMILARITY_THRESHOLD,
    MIN_TEXT_LENGTH,
    DEFAULT_WORKERS
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize spaCy
try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
except OSError:
    logger.error(f"Could not load spaCy model: {SPACY_MODEL}. "
                 f"Run 'python -m spacy download {SPACY_MODEL}' to install it.")
    # Fallback to a simpler model if available, or set to None
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.warning("Falling back to en_core_web_sm model.")
    except OSError:
        logger.warning("No spaCy model available. Vector similarity features will be disabled.")
        nlp = None


class SentimentEngine:
    """
    Engine for analyzing sentiment in tweets using multiple techniques.
    """
    
    def __init__(self, use_spacy: bool = True, n_workers: Optional[int] = DEFAULT_WORKERS):
        """
        Initialize the sentiment engine.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP features
            n_workers: Number of worker threads for parallel processing
                       (None = use CPU count)
        """
        self.use_spacy = use_spacy and nlp is not None
        self.n_workers = n_workers or os.cpu_count() or 1
        
        # Combine domain-specific lexicons with general ones
        self.positive_words = POSITIVE_WORDS.union(TECH_POSITIVE_WORDS)
        self.negative_words = NEGATIVE_WORDS.union(TECH_NEGATIVE_WORDS)
        
        # Preload spaCy docs for centroids to improve performance
        if self.use_spacy:
            self.positive_centroid_docs = [nlp(word) for word in POSITIVE_CENTROIDS]
            self.negative_centroid_docs = [nlp(word) for word in NEGATIVE_CENTROIDS]
    
    def analyze_tweet(self, tweet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment of a single tweet.
        
        Args:
            tweet: Tweet dictionary with id and text
            
        Returns:
            Dictionary with sentiment analysis results or None if processing failed
        """
        try:
            text = tweet["text"]
            cleaned_text = preprocess_tweet(text)
            
            # Skip if cleaned text is too short
            if len(cleaned_text) < MIN_TEXT_LENGTH:
                return None
            
            # Get sentiment score
            if self.use_spacy:
                sentiment_score, explanation = self._analyze_with_spacy(cleaned_text)
            else:
                sentiment_score, explanation = self._analyze_with_lexicon(cleaned_text)
            
            # Determine sentiment
            sentiment = "positive" if sentiment_score >= 0 else "negative"
            
            return {
                "id": tweet["id"],
                "text": text,
                "cleaned_text": cleaned_text,
                "sentiment_score": sentiment_score,
                "sentiment": sentiment,
                "explanation": explanation
            }
        except Exception as e:
            logger.exception(f"Error analyzing tweet {tweet.get('id', 'unknown')}: {e}")
            return None
    
    def analyze_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple tweets in parallel.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        if not tweets:
            logger.warning("No tweets to analyze")
            return []
        
        logger.info(f"Analyzing sentiment for {len(tweets)} tweets...")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self.analyze_tweet, tweets))
        
        # Filter out None results
        analyzed_tweets = [tweet for tweet in results if tweet is not None]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Sentiment analysis completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully analyzed {len(analyzed_tweets)} of {len(tweets)} tweets")
        
        return analyzed_tweets
    
    def _analyze_with_spacy(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze sentiment using spaCy's word vectors and lexicon.
        
        Args:
            text: Cleaned tweet text
            
        Returns:
            Tuple of (sentiment_score, explanation_dict)
        """
        # Process the text with spaCy
        doc = nlp(text.lower())
        
        # Filter out stop words and punctuation
        tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        
        # Initialize counters
        positive_count = 0
        negative_count = 0
        
        # Track words contributing to sentiment
        explanation = {
            "positive_words": [],
            "negative_words": [],
            "similar_to_positive": [],
            "similar_to_negative": []
        }
        
        # Count positive and negative words from lexicon
        for token in tokens:
            if token.text in self.positive_words:
                positive_count += 1
                explanation["positive_words"].append(token.text)
            elif token.text in self.negative_words:
                negative_count += 1
                explanation["negative_words"].append(token.text)
            # Use vector similarity for words not in lexicon
            elif token.has_vector:
                # Compare with positive centroids
                max_positive_sim = max(
                    token.similarity(centroid) for centroid in self.positive_centroid_docs
                )
                
                # Compare with negative centroids
                max_negative_sim = max(
                    token.similarity(centroid) for centroid in self.negative_centroid_docs
                )
                
                # Add to sentiment if similarity is above threshold
                if max_positive_sim > SIMILARITY_THRESHOLD and max_positive_sim > max_negative_sim:
                    positive_count += max_positive_sim
                    explanation["similar_to_positive"].append(
                        {"word": token.text, "similarity": float(max_positive_sim)}
                    )
                elif max_negative_sim > SIMILARITY_THRESHOLD and max_negative_sim > max_positive_sim:
                    negative_count += max_negative_sim
                    explanation["similar_to_negative"].append(
                        {"word": token.text, "similarity": float(max_negative_sim)}
                    )
        
        # Check for emoticons
        for emoticon in POSITIVE_EMOTICONS:
            if emoticon in text:
                positive_count += 1
                explanation["positive_words"].append(emoticon)
        
        for emoticon in NEGATIVE_EMOTICONS:
            if emoticon in text:
                negative_count += 1
                explanation["negative_words"].append(emoticon)
        
        # Calculate final score
        total = positive_count + negative_count
        
        # Add summary to explanation
        explanation["summary"] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total
        }
        
        if total == 0:
            return 0.0, explanation
            
        score = (positive_count - negative_count) / total
        return score, explanation
    
    def _analyze_with_lexicon(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze sentiment using only lexicon (no word vectors).
        Used when spaCy is not available.
        
        Args:
            text: Cleaned tweet text
            
        Returns:
            Tuple of (sentiment_score, explanation_dict)
        """
        # Tokenize and lowercase
        tokens = tokenize_with_regex(text.lower())
        
        # Initialize counters and explanations
        positive_count = 0
        negative_count = 0
        
        explanation = {
            "positive_words": [],
            "negative_words": []
        }
        
        # Count sentiment words
        for token in tokens:
            if token in self.positive_words:
                positive_count += 1
                explanation["positive_words"].append(token)
            elif token in self.negative_words:
                negative_count += 1
                explanation["negative_words"].append(token)
        
        # Check for emoticons
        for emoticon in POSITIVE_EMOTICONS:
            if emoticon in text:
                positive_count += 1
                explanation["positive_words"].append(emoticon)
        
        for emoticon in NEGATIVE_EMOTICONS:
            if emoticon in text:
                negative_count += 1
                explanation["negative_words"].append(emoticon)
        
        # Calculate score
        total = positive_count + negative_count
        
        # Add summary to explanation
        explanation["summary"] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total
        }
        
        if total == 0:
            return 0.0, explanation
            
        score = (positive_count - negative_count) / total
        return score, explanation
        
    def get_sentiment_distribution(self, analyzed_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the distribution of sentiment in analyzed tweets.
        
        Args:
            analyzed_tweets: List of analyzed tweet dictionaries
            
        Returns:
            Dictionary with sentiment distribution statistics
        """
        if not analyzed_tweets:
            return {
                "positive_count": 0,
                "negative_count": 0,
                "total_count": 0,
                "positive_percentage": 0.0,
                "negative_percentage": 0.0
            }
            
        positive_count = sum(1 for tweet in analyzed_tweets 
                            if tweet.get("sentiment") == "positive")
        negative_count = sum(1 for tweet in analyzed_tweets 
                            if tweet.get("sentiment") == "negative")
        total_count = len(analyzed_tweets)
        
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total_count,
            "positive_percentage": (positive_count / total_count * 100) if total_count > 0 else 0.0,
            "negative_percentage": (negative_count / total_count * 100) if total_count > 0 else 0.0
        }
        
    def get_common_sentiment_words(self, analyzed_tweets: List[Dict[str, Any]], 
                                  sentiment: str = "positive", limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most common words associated with a particular sentiment.
        
        Args:
            analyzed_tweets: List of analyzed tweet dictionaries
            sentiment: Which sentiment to analyze ("positive" or "negative")
            limit: Maximum number of words to return
            
        Returns:
            List of (word, count) tuples for the most common sentiment words
        """
        if not analyzed_tweets:
            return []
            
        counter = Counter()
        
        for tweet in analyzed_tweets:
            if tweet.get("sentiment") == sentiment and "explanation" in tweet:
                # Count lexicon words
                word_list_key = "positive_words" if sentiment == "positive" else "negative_words"
                if word_list_key in tweet["explanation"]:
                    counter.update(tweet["explanation"][word_list_key])
                
                # Count similar words
                similar_key = "similar_to_positive" if sentiment == "positive" else "similar_to_negative"
                if similar_key in tweet["explanation"]:
                    similar_words = [item["word"] for item in tweet["explanation"][similar_key]]
                    counter.update(similar_words)
        
        return counter.most_common(limit)