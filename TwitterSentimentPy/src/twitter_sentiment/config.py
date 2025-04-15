import os
from pathlib import Path

TWITTER_API_BASE_URL = "https://api.twitter.com/2/tweets/search/recent"
DEFAULT_MAX_RESULTS = 10

ROOT_DIR = Path(__file__).parent.absolute()
DEFAULT_OUTPUT_DIR = Path.cwd() / "output"

DEFAULT_OUTPUT_PATHS = {
    "visualization": str(DEFAULT_OUTPUT_DIR / "sentiment_analysis.png"),
    "wordcloud_positive": str(DEFAULT_OUTPUT_DIR / "positive_wordcloud.png"),
    "wordcloud_negative": str(DEFAULT_OUTPUT_DIR / "negative_wordcloud.png"),
    "hashtag_sentiment": str(DEFAULT_OUTPUT_DIR / "hashtag_sentiment.png"),
    "csv": str(DEFAULT_OUTPUT_DIR / "sentiment_results.csv"),
    "json": str(DEFAULT_OUTPUT_DIR / "sentiment_results.json")
}

SPACY_MODEL = "en_core_web_md"
SIMILARITY_THRESHOLD = 0.6
MIN_TEXT_LENGTH = 3

DEFAULT_WORKERS = None

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)