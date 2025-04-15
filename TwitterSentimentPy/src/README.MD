# Twitter Sentiment Analysis

A comprehensive Python package for analyzing sentiment in tweets using NLP techniques and visualizing the results.

## Features

- **Tweet Fetching**: Collect tweets from Twitter API v2 or use sample data
- **Advanced Sentiment Analysis**: Analyze sentiment using spaCy and lexicon-based approaches
- **Rich Visualizations**: Generate bar charts, word clouds, and comprehensive dashboards
- **Hashtag Analysis**: Break down sentiment by hashtags
- **Performance Optimization**: Parallel processing for fast analysis of large datasets
- **Multiple Export Formats**: Save results as CSV, JSON, and PNG visualizations

## Installation

### Prerequisites

- Python 3.12
- [spaCy](https://spacy.io/) for NLP features
- Twitter API access (optional)

### Install from Source

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -e .
```

If you want to use the advanced NLP features, download the spaCy model:

```bash
python -m spacy download en_core_web_md
```

## Usage

### Command Line Interface

```bash
# Basic analysis
analyze-tweets "python programming"

# Analyze with more tweets
analyze-tweets "python programming" -n 200

# Perform hashtag analysis
analyze-tweets "python programming" --hashtag-analysis

# Specify output directory
analyze-tweets "python programming" -o ./results
```

### Python API

```python
from twitter_sentiment import TwitterSentimentAnalyzer

# Initialize the analyzer
analyzer = TwitterSentimentAnalyzer(
    bearer_token="YOUR_TWITTER_API_TOKEN"  # Optional
)

# Perform sentiment analysis
analyzed_tweets, summary = analyzer.analyze(
    query="python programming",
    max_results=100,
    save_results=True,
    create_visualizations=True
)

# Print summary
print(f"Positive: {summary['positive_percentage']:.1f}%")
print(f"Negative: {summary['negative_percentage']:.1f}%")
print(f"Total analyzed: {summary['total_count']}")
```

## Project Structure

```
twitter_sentiment_analysis/
├── src/
│   └── twitter_sentiment/         # Main package
│       ├── core/                  # Core functionality
│       │   ├── analyzer.py        # Main analyzer class
│       │   ├── data_fetcher.py    # Twitter API integration
│       │   └── sentiment_engine.py # Sentiment analysis
│       ├── utils/                 # Utility functions
│       ├── visualization/         # Visualization components
│       ├── data/                  # Lexicons and sample data
│       └── config.py              # Configuration
├── scripts/                       # Command-line scripts
└── README.md                      # This file
```

## Example Outputs

The package generates various visualizations:

1. **Sentiment Distribution**: Bar chart showing positive vs. negative tweets
2. **Word Clouds**: Visual representation of common words in positive and negative tweets
3. **Hashtag Analysis**: Breakdown of sentiment by popular hashtags
4. **Dashboard**: Comprehensive visualization combining multiple charts

## Configuration

You can configure the analyzer by setting environment variables or passing parameters:

- `TWITTER_BEARER_TOKEN`: Your Twitter API bearer token
- Output directories and filenames can be customized in `config.py` or at runtime

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [spaCy](https://spacy.io/) for NLP processing
- [matplotlib](https://matplotlib.org/) and [wordcloud](https://github.com/amueller/word_cloud) for visualizations
