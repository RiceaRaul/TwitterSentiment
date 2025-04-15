import os
import requests
from typing import List, Dict, Any, Optional
import logging

from twitter_sentiment.config import TWITTER_API_BASE_URL
from twitter_sentiment.data.sample_tweets import generate_sample_tweets

# Set up logging
logger = logging.getLogger(__name__)


class TwitterDataFetcher:
    """Class for fetching data from Twitter API."""
    
    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize the Twitter data fetcher.
        
        Args:
            bearer_token: Twitter API bearer token. If None, will try to use
                          the TWITTER_BEARER_TOKEN environment variable.
        """
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        if not self.bearer_token:
            logger.warning("No Twitter API bearer token provided. "
                           "Will use sample data when fetching tweets.")
    
    def fetch_tweets(self, query: str, max_results: int = 10,
                    tweet_fields: str = "id,text,created_at") -> List[Dict[str, Any]]:
        """
        Fetch tweets from the Twitter API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            tweet_fields: Comma-separated list of tweet fields to include
            
        Returns:
            List of tweet dictionaries
        """
        if not self.bearer_token:
            logger.info("No bearer token available, using sample data.")
            return generate_sample_tweets(max_results)
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": tweet_fields
        }
        
        logger.info(f"Fetching tweets for query: '{query}'...")
        
        try:
            response = requests.get(
                TWITTER_API_BASE_URL,
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Error: API returned status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return generate_sample_tweets(max_results)
            
            data = response.json()
            
            if "data" not in data or not data["data"]:
                logger.warning("No tweets found, using sample data")
                return generate_sample_tweets(max_results)
            
            logger.info(f"Successfully fetched {len(data['data'])} tweets")
            return data["data"]
        
        except Exception as e:
            logger.exception(f"Error fetching tweets: {e}")
            return generate_sample_tweets(max_results)
    
    def fetch_user_tweets(self, user_id: str, max_results: int = 10,
                         tweet_fields: str = "id,text,created_at") -> List[Dict[str, Any]]:
        """
        Fetch tweets from a specific user.
        
        Args:
            user_id: Twitter user ID
            max_results: Maximum number of results to return
            tweet_fields: Comma-separated list of tweet fields to include
            
        Returns:
            List of tweet dictionaries
        """
        if not self.bearer_token:
            logger.info("No bearer token available, using sample data.")
            return generate_sample_tweets(max_results)
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        params = {
            "max_results": max_results,
            "tweet.fields": tweet_fields
        }
        
        url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        
        logger.info(f"Fetching tweets for user ID: {user_id}...")
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Error: API returned status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return generate_sample_tweets(max_results)
            
            data = response.json()
            
            if "data" not in data or not data["data"]:
                logger.warning("No tweets found, using sample data")
                return generate_sample_tweets(max_results)
            
            logger.info(f"Successfully fetched {len(data['data'])} tweets")
            return data["data"]
        
        except Exception as e:
            logger.exception(f"Error fetching user tweets: {e}")
            return generate_sample_tweets(max_results)