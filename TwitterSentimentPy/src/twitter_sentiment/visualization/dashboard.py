import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from twitter_sentiment.utils.text_processing import extract_hashtags
from twitter_sentiment.visualization.word_clouds import generate_wordcloud

# Set up logging
logger = logging.getLogger(__name__)


def create_dashboard(
    analyzed_tweets: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Twitter Sentiment Analysis Dashboard",
    show_plot: bool = False
) -> None:
    """
    Create a comprehensive dashboard visualization for Twitter sentiment analysis.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        summary: Dictionary with sentiment summary statistics
        output_path: Path to save the dashboard image
        title: Title for the dashboard
        show_plot: Whether to display the plot interactively
    """
    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sentiment analysis dashboard to: {output_path}")
    
    if not analyzed_tweets:
        logger.warning("No tweets to visualize in dashboard")
        return
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    _create_sentiment_barchart(ax1, summary)
    
    ax2 = fig.add_subplot(gs[0, 1])
    _create_wordcloud_panel(
        ax2, 
        analyzed_tweets, 
        sentiment="positive", 
        title="Positive Tweets - Word Cloud"
    )
    
    ax3 = fig.add_subplot(gs[1, 0])
    _create_wordcloud_panel(
        ax3, 
        analyzed_tweets, 
        sentiment="negative", 
        title="Negative Tweets - Word Cloud"
    )
    
    ax4 = fig.add_subplot(gs[1, 1])
    _create_hashtag_panel(ax4, analyzed_tweets)
    
    fig.suptitle(title, fontsize=16)
    fig.text(
        0.5, 0.01,
        f'Total analyzed: {summary["total_count"]} tweets',
        ha='center',
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    logger.info(f"Dashboard saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def _create_sentiment_barchart(ax, summary: Dict[str, Any]) -> None:
    """Helper function to create sentiment bar chart for dashboard."""
    labels = ['Positive', 'Negative']
    values = [summary['positive_count'], summary['negative_count']]
    percentages = [summary['positive_percentage'], summary['negative_percentage']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, values, color=colors)
    
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_xlabel('Sentiment', fontsize=10)
    ax.set_ylabel('Number of Tweets', fontsize=10)
    ax.set_title('Sentiment Distribution', fontsize=12)


def _create_wordcloud_panel(
    ax, 
    analyzed_tweets: List[Dict[str, Any]],
    sentiment: str = "positive",
    title: str = "Word Cloud"
) -> None:
    """Helper function to create word cloud panel for dashboard."""

    filtered_text = " ".join([
        tweet["cleaned_text"] for tweet in analyzed_tweets
        if tweet.get("sentiment") == sentiment
    ])
    
    if filtered_text:
        temp_path = Path(f"temp_{sentiment}_wordcloud.png")
        
        try:

            colormap = "Greens" if sentiment == "positive" else "Reds"
            generate_wordcloud(
                filtered_text,
                temp_path,
                colormap=colormap,
                show_plot=False
            )
            
            if temp_path.exists():
                img = plt.imread(temp_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title, fontsize=12)
                
                os.remove(temp_path)
            else:
                ax.text(0.5, 0.5, "Failed to generate word cloud", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}")
            ax.text(0.5, 0.5, "Error generating word cloud", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, f"No {sentiment} tweets to display", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def _create_hashtag_panel(ax, analyzed_tweets: List[Dict[str, Any]]) -> None:
    """Helper function to create hashtag bar chart for dashboard."""

    hashtag_counts = Counter()
    
    for tweet in analyzed_tweets:
        hashtags = extract_hashtags(tweet["text"])
        hashtag_counts.update(hashtags)
    
    top_hashtags = hashtag_counts.most_common(5)
    
    if top_hashtags:
        labels = [f"#{label}" for label, _ in top_hashtags]
        values = [count for _, count in top_hashtags]
        
        bars = ax.barh(labels, values, color='skyblue')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.1,
                bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        ax.set_xlabel('Count', fontsize=10)
        ax.set_title('Top Hashtags', fontsize=12)
    else:
        ax.text(0.5, 0.5, "No hashtags found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def create_sentiment_breakdown_dashboard(
    analyzed_tweets: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Sentiment Analysis Breakdown",
    show_plot: bool = False
) -> None:
    """
    Create a dashboard focused on sentiment breakdown by various factors.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        output_path: Path to save the dashboard image
        title: Title for the dashboard
        show_plot: Whether to display the plot interactively
    """

    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sentiment breakdown dashboard to: {output_path}")
    
    if not analyzed_tweets:
        logger.warning("No tweets to visualize in breakdown dashboard")
        return
    
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    _create_sentiment_words_panel(
        ax1, 
        analyzed_tweets, 
        sentiment="positive", 
        title="Top Positive Words"
    )
    
    ax2 = fig.add_subplot(gs[0, 1])
    _create_sentiment_words_panel(
        ax2, 
        analyzed_tweets, 
        sentiment="negative", 
        title="Top Negative Words"
    )
    
    ax3 = fig.add_subplot(gs[1, 0])
    _create_sentiment_by_hashtag_panel(ax3, analyzed_tweets)
    
    ax4 = fig.add_subplot(gs[1, 1])
    _create_sentiment_score_histogram(ax4, analyzed_tweets)
    
    ax5 = fig.add_subplot(gs[2, 0])
    _create_example_tweets_panel(
        ax5, 
        analyzed_tweets, 
        sentiment="positive", 
        title="Example Positive Tweets", 
        max_tweets=5
    )
    
    ax6 = fig.add_subplot(gs[2, 1])
    _create_example_tweets_panel(
        ax6, 
        analyzed_tweets, 
        sentiment="negative", 
        title="Example Negative Tweets", 
        max_tweets=5
    )
    
    fig.suptitle(title, fontsize=16)
    fig.text(
        0.5, 0.01,
        f'Total analyzed: {len(analyzed_tweets)} tweets',
        ha='center',
        fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    logger.info(f"Sentiment breakdown dashboard saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def _create_sentiment_words_panel(
    ax, 
    analyzed_tweets: List[Dict[str, Any]],
    sentiment: str = "positive",
    title: str = "Top Words",
    max_words: int = 10
) -> None:
    """Helper function to create panel showing top words for a sentiment."""

    word_counts = Counter()
    
    for tweet in analyzed_tweets:
        if tweet.get("sentiment") == sentiment and "explanation" in tweet:
            word_list_key = "positive_words" if sentiment == "positive" else "negative_words"
            
            if word_list_key in tweet["explanation"]:
                word_counts.update(tweet["explanation"][word_list_key])
    
    if not word_counts:
        for tweet in analyzed_tweets:
            if tweet.get("sentiment") == sentiment:
                words = tweet["cleaned_text"].lower().split()
                word_counts.update(words)
    
    top_words = word_counts.most_common(max_words)
    
    if top_words:
        words = [word for word, _ in top_words]
        counts = [count for _, count in top_words]
        
        colors = ['#2ecc71'] * len(words) if sentiment == "positive" else ['#e74c3c'] * len(words)
        
        y_pos = range(len(words))
        ax.barh(y_pos, counts, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        
        for i, count in enumerate(counts):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
        
        ax.set_xlabel('Count', fontsize=10)
        ax.set_title(title, fontsize=12)
    else:
        ax.text(0.5, 0.5, f"No {sentiment} words found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def _create_sentiment_by_hashtag_panel(
    ax, 
    analyzed_tweets: List[Dict[str, Any]], 
    max_hashtags: int = 5
) -> None:
    """Helper function to create panel showing sentiment by hashtag."""
    hashtag_data = {}
    
    for tweet in analyzed_tweets:
        hashtags = extract_hashtags(tweet["text"])
        sentiment = tweet.get("sentiment")
        
        for hashtag in hashtags:
            if hashtag not in hashtag_data:
                hashtag_data[hashtag] = {
                    "total": 0,
                    "positive": 0,
                    "negative": 0
                }
            
            hashtag_data[hashtag]["total"] += 1
            
            if sentiment == "positive":
                hashtag_data[hashtag]["positive"] += 1
            else:
                hashtag_data[hashtag]["negative"] += 1
    
    # Sort by total count and take top N
    top_hashtags = sorted(
        hashtag_data.items(), 
        key=lambda x: x[1]["total"], 
        reverse=True
    )[:max_hashtags]
    
    if top_hashtags:
        hashtags = [f"#{h}" for h, _ in top_hashtags]
        positive_counts = [data["positive"] for _, data in top_hashtags]
        negative_counts = [data["negative"] for _, data in top_hashtags]
        total_counts = [data["total"] for _, data in top_hashtags]
        
        pos_percentages = [
            (pos / total) * 100 for pos, total in zip(positive_counts, total_counts)
        ]
        neg_percentages = [
            (neg / total) * 100 for neg, total in zip(negative_counts, total_counts)
        ]
        
        ax.barh(hashtags, pos_percentages, color='#2ecc71', label='Positive')
        ax.barh(hashtags, neg_percentages, left=pos_percentages, color='#e74c3c', label='Negative')
        
        for i, (pos, neg) in enumerate(zip(pos_percentages, neg_percentages)):
            if pos > 10:
                ax.text(
                    pos/2, 
                    i, 
                    f'{pos:.0f}%', 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=9
                )
            
            if neg > 10: 
                ax.text(
                    pos + neg/2, 
                    i, 
                    f'{neg:.0f}%', 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=9
                )
        
        for i, count in enumerate(total_counts):
            ax.text(
                101,
                i, 
                f'{count} tweets', 
                va='center', 
                fontsize=8
            )
        
        ax.set_xlim(0, 120)
        ax.set_xlabel('Percentage', fontsize=10)
        ax.set_title('Sentiment by Hashtag', fontsize=12)
        ax.legend(loc='lower right')
    else:
        ax.text(0.5, 0.5, "No hashtags found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def _create_sentiment_score_histogram(
    ax, 
    analyzed_tweets: List[Dict[str, Any]]
) -> None:
    """Helper function to create histogram of sentiment scores."""
    scores = [tweet.get("sentiment_score", 0) for tweet in analyzed_tweets]
    
    if scores:
        ax.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Sentiment Score', fontsize=10)
        ax.set_ylabel('Number of Tweets', fontsize=10)
        ax.set_title('Sentiment Score Distribution', fontsize=12)
        
        ax.text(
            -0.8, 
            ax.get_ylim()[1] * 0.9, 
            'Negative', 
            ha='center', 
            va='center',
            fontsize=9,
            color='#e74c3c'
        )
        
        ax.text(
            0.8, 
            ax.get_ylim()[1] * 0.9, 
            'Positive', 
            ha='center', 
            va='center',
            fontsize=9,
            color='#2ecc71'
        )
    else:
        ax.text(0.5, 0.5, "No sentiment scores available", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def _create_example_tweets_panel(
    ax, 
    analyzed_tweets: List[Dict[str, Any]],
    sentiment: str = "positive",
    title: str = "Example Tweets",
    max_tweets: int = 5
) -> None:
    """Helper function to create panel showing example tweets."""

    filtered_tweets = [
        tweet for tweet in analyzed_tweets
        if tweet.get("sentiment") == sentiment
    ]
    
    sorted_tweets = sorted(
        filtered_tweets,
        key=lambda x: abs(x.get("sentiment_score", 0)),
        reverse=True
    )
    
    example_tweets = sorted_tweets[:max_tweets]
    
    if example_tweets:
        ax.axis('off')
        ax.set_title(title, fontsize=12)
        
        text_content = ""
        for i, tweet in enumerate(example_tweets):

            text = tweet["text"]
            if len(text) > 100:
                text = text[:97] + "..."
            
            text_content += f"{i+1}. {text}\n\n"
        
        ax.text(
            0.02, 0.98,
            text_content,
            va='top',
            ha='left',
            transform=ax.transAxes,
            fontsize=9,
            wrap=True
        )
    else:
        ax.text(0.5, 0.5, f"No {sentiment} tweets found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')