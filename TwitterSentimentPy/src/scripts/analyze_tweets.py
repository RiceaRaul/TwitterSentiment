import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from twitter_sentiment import TwitterSentimentAnalyzer


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Twitter Sentiment Analysis Tool'
    )
    
    parser.add_argument(
        'query',
        help='Search query for Twitter'
    )
    
    parser.add_argument(
        '-n', '--max-results',
        type=int,
        default=100,
        help='Maximum number of tweets to fetch (default: 100)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '-t', '--token',
        help='Twitter API bearer token (can also use TWITTER_BEARER_TOKEN env variable)'
    )
    
    parser.add_argument(
        '--no-spacy',
        action='store_true',
        help='Disable spaCy for NLP features (faster but less accurate)'
    )
    
    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Save results but do not create visualizations'
    )
    
    parser.add_argument(
        '--hashtag-analysis',
        action='store_true',
        help='Perform additional hashtag-based analysis'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Get token from args or environment
    token = args.token or os.environ.get('TWITTER_BEARER_TOKEN')
    
    # Initialize the analyzer
    analyzer = TwitterSentimentAnalyzer(
        bearer_token=token,
        use_spacy=not args.no_spacy,
        output_dir=args.output_dir
    )
    
    try:
        # Perform analysis
        if args.hashtag_analysis:
            analyzer.analyze_by_hashtag(
                args.query,
                max_results=args.max_results,
                save_results=True
            )
        else:
            analyzer.analyze(
                args.query,
                max_results=args.max_results,
                save_results=True,
                create_visualizations=not args.save_only
            )
            
        print("\nAnalysis completed successfully. Check output directory for results.")
            
    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()