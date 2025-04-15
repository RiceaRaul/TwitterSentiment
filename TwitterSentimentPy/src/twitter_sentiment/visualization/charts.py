import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Set up logging
logger = logging.getLogger(__name__)


def create_sentiment_bar_chart(
    summary: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Sentiment Distribution",
    show_plot: bool = False
) -> None:
    """
    Create a bar chart of sentiment distribution.
    
    Args:
        summary: Dictionary with sentiment summary statistics
        output_path: Path to save the chart image
        title: Title for the chart
        show_plot: Whether to display the plot interactively
    """
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sentiment bar chart to: {output_path}")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Data for the chart
    labels = ['Positive', 'Negative']
    values = [summary['positive_count'], summary['negative_count']]
    percentages = [summary['positive_percentage'], summary['negative_percentage']]
    colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative
    
    # Create bar chart
    bars = plt.bar(labels, values, color=colors)
    
    # Add percentages above bars
    for bar, percentage in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add labels and title
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Add total count as text
    plt.figtext(
        0.5, 0.01,
        f'Total analyzed: {summary["total_count"]} tweets',
        ha='center',
        fontsize=12
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_path)
    logger.info(f"Bar chart saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_hashtag_bar_chart(
    hashtag_counts: List[Tuple[str, int]],
    output_path: Union[str, Path],
    title: str = "Top Hashtags",
    color: str = 'skyblue',
    max_hashtags: int = 10,
    show_plot: bool = False
) -> None:
    """
    Create a bar chart of top hashtags.
    
    Args:
        hashtag_counts: List of (hashtag, count) tuples
        output_path: Path to save the chart image
        title: Title for the chart
        color: Color for the bars
        max_hashtags: Maximum number of hashtags to include
        show_plot: Whether to display the plot interactively
    """
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating hashtag bar chart to: {output_path}")
    
    # Take the top N hashtags
    top_hashtags = hashtag_counts[:max_hashtags]
    
    if not top_hashtags:
        logger.warning("No hashtags to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Extract hashtags and counts
    hashtags = [f"#{h}" for h, _ in top_hashtags]
    counts = [c for _, c in top_hashtags]
    
    # Create bar chart (horizontal for better readability with long hashtags)
    bars = plt.barh(hashtags, counts, color=color)
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{int(width)}',
            ha='left',
            va='center',
            fontsize=10
        )
    
    # Add labels and title
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Hashtags', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    logger.info(f"Hashtag bar chart saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_sentiment_by_hashtag_chart(
    hashtag_sentiment: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Sentiment by Hashtag",
    max_hashtags: int = 5,
    show_plot: bool = False
) -> None:
    """
    Create a chart showing sentiment by hashtag.
    
    Args:
        hashtag_sentiment: List of dictionaries with hashtag sentiment information
        output_path: Path to save the chart image
        title: Title for the chart
        max_hashtags: Maximum number of hashtags to include
        show_plot: Whether to display the plot interactively
    """
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sentiment by hashtag chart to: {output_path}")
    
    if not hashtag_sentiment:
        logger.warning("No hashtag sentiment data to visualize")
        return
    
    # Sort by tweet count and take top N
    sorted_data = sorted(
        hashtag_sentiment,
        key=lambda x: x['tweet_count'],
        reverse=True
    )[:max_hashtags]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Extract data
    hashtags = [f"#{item['hashtag']}" for item in sorted_data]
    positive_pct = [item['positive_percentage'] for item in sorted_data]
    negative_pct = [100 - pct for pct in positive_pct]
    tweet_counts = [item['tweet_count'] for item in sorted_data]
    
    # Set up the bar chart
    x = np.arange(len(hashtags))
    width = 0.35
    
    # Create stacked bar chart
    plt.bar(x, positive_pct, width, label='Positive', color='#2ecc71')
    plt.bar(x, negative_pct, width, bottom=positive_pct, label='Negative', color='#e74c3c')
    
    # Add tweet count labels
    for i, count in enumerate(tweet_counts):
        plt.text(
            i,
            110,  # Position above the bars
            f'{count} tweets',
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=90
        )
    
    # Add percentage labels
    for i, pct in enumerate(positive_pct):
        if pct > 10:  # Only add label if there's enough space
            plt.text(
                i,
                pct/2,  # Middle of the positive section
                f'{pct:.1f}%',
                ha='center',
                va='center',
                fontsize=9,
                color='white'
            )
        
        if negative_pct[i] > 10:  # Only add label if there's enough space
            plt.text(
                i,
                pct + negative_pct[i]/2,  # Middle of the negative section
                f'{negative_pct[i]:.1f}%',
                ha='center',
                va='center',
                fontsize=9,
                color='white'
            )
    
    # Add labels and title
    plt.xlabel('Hashtags', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(x, hashtags, rotation=45, ha='right')
    plt.ylim(0, 120)  # Make room for tweet count labels
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    logger.info(f"Sentiment by hashtag chart saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_time_series_chart(
    analyzed_tweets: List[Dict[str, Any]],
    output_path: Union[str, Path],
    time_field: str = "created_at",
    interval: str = "day",
    title: str = "Sentiment Over Time",
    show_plot: bool = False
) -> None:
    """
    Create a time series chart of sentiment.
    
    Args:
        analyzed_tweets: List of analyzed tweet dictionaries
        output_path: Path to save the chart image
        time_field: Field containing timestamp
        interval: Time interval for grouping ('hour', 'day', 'week', 'month')
        title: Title for the chart
        show_plot: Whether to display the plot interactively
    """
    # This function requires the tweets to have a timestamp field
    # For simplicity, we'll assume it's in ISO format (e.g., '2021-01-01T12:00:00Z')
    import datetime
    import dateutil.parser
    
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating time series chart to: {output_path}")
    
    # Check if tweets have timestamp field
    if not analyzed_tweets or time_field not in analyzed_tweets[0]:
        logger.warning(f"Tweets do not have '{time_field}' field for time series analysis")
        return
    
    # Parse timestamps and group by interval
    positive_by_time = {}
    negative_by_time = {}
    
    for tweet in analyzed_tweets:
        try:
            # Parse timestamp
            timestamp = dateutil.parser.parse(tweet[time_field])
            
            # Truncate timestamp to the specified interval
            if interval == 'hour':
                key = timestamp.replace(minute=0, second=0, microsecond=0)
            elif interval == 'day':
                key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif interval == 'week':
                # Get the start of the week (Monday)
                key = timestamp - datetime.timedelta(days=timestamp.weekday())
                key = key.replace(hour=0, minute=0, second=0, microsecond=0)
            elif interval == 'month':
                key = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                # Default to day
                key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Count by sentiment
            if tweet['sentiment'] == 'positive':
                positive_by_time[key] = positive_by_time.get(key, 0) + 1
            else:
                negative_by_time[key] = negative_by_time.get(key, 0) + 1
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing timestamp from tweet: {e}")
    
    if not positive_by_time and not negative_by_time:
        logger.warning("No valid timestamps found for time series analysis")
        return
    
    # Get all unique timestamps
    all_times = sorted(set(list(positive_by_time.keys()) + list(negative_by_time.keys())))
    
    # Fill in missing values with zeros
    positive_counts = [positive_by_time.get(t, 0) for t in all_times]
    negative_counts = [negative_by_time.get(t, 0) for t in all_times]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot the data
    plt.plot(all_times, positive_counts, 'g-', label='Positive', marker='o')
    plt.plot(all_times, negative_counts, 'r-', label='Negative', marker='x')
    
    # Format date axis
    plt.gcf().autofmt_xdate()
    
    # Add labels and title
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    logger.info(f"Time series chart saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()