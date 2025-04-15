import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Set

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS

# Set up logging
logger = logging.getLogger(__name__)


def generate_wordcloud(
    text: str,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
    colormap: str = 'viridis',
    max_words: int = 200,
    stopwords: Optional[Set[str]] = None,
    show_plot: bool = False
) -> None:
    """
    Generate a word cloud from text.
    
    Args:
        text: Text to generate word cloud from
        output_path: Path to save the word cloud image
        title: Title for the word cloud (optional)
        width: Width of the word cloud image
        height: Height of the word cloud image
        background_color: Background color of the word cloud
        colormap: Matplotlib colormap to use
        max_words: Maximum number of words to include
        stopwords: Set of stopwords to exclude (uses spaCy default if None)
        show_plot: Whether to display the plot interactively
    """
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating word cloud to: {output_path}")
    
    # Use default stopwords if none provided
    if stopwords is None:
        stopwords = STOP_WORDS
    
    # Check if there's text to process
    if not text.strip():
        logger.warning("No text to generate word cloud from")
        return
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=stopwords,
        max_words=max_words,
        min_font_size=10,
        max_font_size=None,  # Auto-scale
        random_state=42  # For reproducibility
    ).generate(text)
    
    # Create plot
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if title:
        plt.title(title, fontsize=16)
    
    # Save the image
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight')
    logger.info(f"Word cloud saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_sentiment_wordclouds(
    analyzed_tweets: List[Dict[str, Any]],
    positive_output_path: Union[str, Path],
    negative_output_path: Union[str, Path],
    additional_stopwords: Optional[Set[str]] = None
) -> None:
    """
    Generate separate word clouds for positive and negative tweets.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        positive_output_path: Path to save the positive word cloud image
        negative_output_path: Path to save the negative word cloud image
        additional_stopwords: Additional stopwords to exclude
    """
    logger.info("Generating sentiment word clouds...")
    
    # Combine default stopwords with additional ones
    stopwords = set(STOP_WORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)
    
    # Separate positive and negative tweets
    positive_text = " ".join([
        tweet["cleaned_text"] for tweet in analyzed_tweets
        if tweet.get("sentiment") == "positive"
    ])
    
    negative_text = " ".join([
        tweet["cleaned_text"] for tweet in analyzed_tweets
        if tweet.get("sentiment") == "negative"
    ])
    
    # Generate positive word cloud
    if positive_text:
        generate_wordcloud(
            positive_text,
            positive_output_path,
            title="Positive Sentiment Word Cloud",
            colormap="Greens",
            stopwords=stopwords
        )
    else:
        logger.warning("No positive tweets to generate word cloud from")
    
    # Generate negative word cloud
    if negative_text:
        generate_wordcloud(
            negative_text,
            negative_output_path,
            title="Negative Sentiment Word Cloud",
            colormap="Reds",
            stopwords=stopwords
        )
    else:
        logger.warning("No negative tweets to generate word cloud from")


def generate_custom_word_clouds(
    analyzed_tweets: List[Dict[str, Any]],
    criteria_fn,
    output_path: Union[str, Path],
    title: str,
    colormap: str = "viridis",
) -> None:
    """
    Generate a word cloud based on custom criteria.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        criteria_fn: Function that takes a tweet and returns True if it should be included
        output_path: Path to save the word cloud image
        title: Title for the word cloud
        colormap: Matplotlib colormap to use
    """
    # Filter tweets based on the criteria function
    filtered_tweets = [tweet for tweet in analyzed_tweets if criteria_fn(tweet)]
    
    if not filtered_tweets:
        logger.warning(f"No tweets match the criteria for '{title}' word cloud")
        return
    
    # Join all the text
    text = " ".join([tweet["cleaned_text"] for tweet in filtered_tweets])
    
    # Generate the word cloud
    generate_wordcloud(
        text,
        output_path,
        title=title,
        colormap=colormap
    )


def extract_key_terms(
    analyzed_tweets: List[Dict[str, Any]], 
    min_frequency: int = 2
) -> Dict[str, int]:
    """
    Extract key terms from analyzed tweets.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        min_frequency: Minimum frequency for a term to be included
        
    Returns:
        Dictionary of terms and their frequencies
    """
    # This is a simple implementation - for more advanced term extraction,
    # consider using TF-IDF or other NLP techniques
    
    # Combine all text
    all_text = " ".join([tweet["cleaned_text"] for tweet in analyzed_tweets])
    
    # Tokenize and count
    words = all_text.lower().split()
    
    # Remove stopwords
    words = [word for word in words if word.lower() not in STOP_WORDS and len(word) > 2]
    
    # Count frequencies
    term_counts = {}
    for word in words:
        term_counts[word] = term_counts.get(word, 0) + 1
    
    # Filter by minimum frequency
    key_terms = {term: count for term, count in term_counts.items() 
                if count >= min_frequency}
    
    return key_terms