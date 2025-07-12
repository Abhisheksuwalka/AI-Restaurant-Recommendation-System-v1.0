import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        cleaned_text = self.clean_text(text)
        blob = TextBlob(cleaned_text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert to sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        cleaned_text = self.clean_text(text)
        scores = self.sia.polarity_scores(cleaned_text)
        
        # Determine sentiment based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_sentiment(self, text):
        """Main sentiment analysis method - combines TextBlob and VADER"""
        textblob_result = self.analyze_sentiment_textblob(text)
        vader_result = self.analyze_sentiment_vader(text)
        
        # Use VADER as primary, TextBlob as secondary
        return {
            'sentiment': vader_result['sentiment'],
            'confidence': abs(vader_result['compound']),
            'polarity': textblob_result['polarity'],
            'compound': vader_result['compound'],
            'scores': {
                'positive': vader_result['positive'],
                'negative': vader_result['negative'],
                'neutral': vader_result['neutral']
            }
        }
    
    def analyze_reviews_batch(self, reviews_df):
        """Analyze sentiment for a batch of reviews"""
        results = []
        
        for _, row in reviews_df.iterrows():
            textblob_result = self.analyze_sentiment_textblob(row['review_text'])
            vader_result = self.analyze_sentiment_vader(row['review_text'])
            
            # Combine results
            result = {
                'user_id': row['user_id'],
                'restaurant_id': row['restaurant_id'],
                'review_text': row['review_text'],
                'rating': row['rating'],
                'textblob_sentiment': textblob_result['sentiment'],
                'textblob_polarity': textblob_result['polarity'],
                'vader_sentiment': vader_result['sentiment'],
                'vader_compound': vader_result['compound'],
                'timestamp': row['timestamp']
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_restaurant_sentiment_score(self, reviews_df, restaurant_id):
        """Calculate overall sentiment score for a restaurant"""
        restaurant_reviews = reviews_df[reviews_df['restaurant_id'] == restaurant_id]
        
        if len(restaurant_reviews) == 0:
            return 0.0
        
        # Calculate weighted sentiment score
        sentiment_scores = []
        for _, review in restaurant_reviews.iterrows():
            # Use VADER compound score
            sentiment_scores.append(review['vader_compound'])
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_scores)
        
        # Calculate sentiment distribution
        positive_count = len(restaurant_reviews[restaurant_reviews['vader_sentiment'] == 'positive'])
        negative_count = len(restaurant_reviews[restaurant_reviews['vader_sentiment'] == 'negative'])
        neutral_count = len(restaurant_reviews[restaurant_reviews['vader_sentiment'] == 'neutral'])
        
        total_reviews = len(restaurant_reviews)
        
        return {
            'avg_sentiment': avg_sentiment,
            'positive_ratio': positive_count / total_reviews,
            'negative_ratio': negative_count / total_reviews,
            'neutral_ratio': neutral_count / total_reviews,
            'total_reviews': total_reviews
        }
