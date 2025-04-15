SAMPLE_TWEETS = [
    {
        "id": "1",
        "text": "I love Python and spaCy for NLP! It's amazingly intuitive. #python #nlp"
    },
    {
        "id": "2",
        "text": "This code keeps crashing. Python errors are so frustrating sometimes. #python #annoyed"
    },
    {
        "id": "3",
        "text": "Big news! Python 3.11 released with great performance improvements. Really excited to try it out!"
    },
    {
        "id": "4",
        "text": "Having memory issues with my code again. Debugging for hours. This is awful."
    },
    {
        "id": "5",
        "text": "The community around Python is so helpful. Thanks to everyone who answered my questions!"
    },
    {
        "id": "6",
        "text": "Why is async in Python so hard to understand? Been stuck on this bug for days."
    },
    {
        "id": "7",
        "text": "Python's performance benchmarks are incredible with 3.11! 30% faster than before."
    },
    {
        "id": "8",
        "text": "Conference talk on Python was disappointing. Speaker barely knew the basics."
    },
    {
        "id": "9",
        "text": "Finally fixed that concurrency issue! Python's asyncio is actually helpful once you understand it."
    },
    {
        "id": "10",
        "text": "My application keeps failing in production. Might have to abandon Python for this project."
    },
    {
        "id": "11",
        "text": "Python's error messages are so clear and helpful. Really improving my code."
    },
    {
        "id": "12",
        "text": "The startup times in Python are terrible. Waiting 5 seconds for a simple script. #slow"
    },
    {
        "id": "13",
        "text": "Just converted our backend to Python and development speed increased by 80%! Amazing results!"
    },
    {
        "id": "14",
        "text": "Another null reference in our Java code. Can't wait to port everything to Python and be done with these issues."
    },
    {
        "id": "15",
        "text": "Python's type hints caught another bug that would have been a runtime error. So impressed!"
    }
]


def generate_sample_tweets(count=100):
    """
    Generate a specified number of sample tweets for testing.
    
    Args:
        count: Number of sample tweets to generate
        
    Returns:
        List of tweet dictionaries
    """
    return [
        {
            "id": str(i + 1),
            "text": SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)]["text"]
        }
        for i in range(count)
    ]