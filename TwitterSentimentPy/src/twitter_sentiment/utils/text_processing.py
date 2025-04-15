import re
from typing import List, Set, Optional


def preprocess_tweet(text: str) -> str:
    """
    Clean and preprocess tweet text by removing URLs, mentions,
    hashtags, RT prefix, and special characters.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned and preprocessed text
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove RT prefix
    text = re.sub(r'^RT', '', text)
    
    # Remove special characters and replace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces and trim
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from tweet text.
    
    Args:
        text: Raw tweet text
        
    Returns:
        List of hashtags (without the # symbol)
    """
    hashtag_pattern = re.compile(r'#(\w+)')
    return hashtag_pattern.findall(text.lower())


def extract_mentions(text: str) -> List[str]:
    """
    Extract mentions from tweet text.
    
    Args:
        text: Raw tweet text
        
    Returns:
        List of mentions (without the @ symbol)
    """
    mention_pattern = re.compile(r'@(\w+)')
    return mention_pattern.findall(text.lower())


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from tweet text.
    
    Args:
        text: Raw tweet text
        
    Returns:
        List of URLs
    """
    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.findall(text)


def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens: List of tokens to process
        stopwords: Set of stopwords to remove. If None, a default set will be used.
        
    Returns:
        List of tokens with stopwords removed
    """
    if stopwords is None:
        # A very small default set - normally you'd use a larger set from NLTK or spaCy
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 
                    'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for',
                    'with', 'by', 'about', 'as', 'of', 'from'}
    
    return [token for token in tokens if token.lower() not in stopwords]


def tokenize_simple(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    return text.split()


def tokenize_with_regex(text: str) -> List[str]:
    """
    Tokenize text using regex to handle punctuation and special cases.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # This is a simple regex-based tokenizer
    # For more advanced tokenization, use spaCy or NLTK
    pattern = r'\b\w+\b'
    return re.findall(pattern, text.lower())