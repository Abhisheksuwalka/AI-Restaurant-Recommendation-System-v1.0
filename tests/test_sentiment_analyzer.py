"""
Comprehensive Tests for Sentiment Analyzer

Tests sentiment analysis functionality including individual review analysis,
batch processing, restaurant sentiment aggregation, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from tests.base_test import BaseTestCase, slow_test
from models.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalyzer(BaseTestCase):
    """Test the SentimentAnalyzer class"""
    
    def setUp(self):
        super().setUp()
        self.analyzer = SentimentAnalyzer()
        
        # Create test reviews with known sentiments
        self.test_reviews = [
            "This restaurant is absolutely amazing! The food is delicious and the service is excellent.",
            "The food was okay, nothing special but not bad either.",
            "Terrible experience! The food was awful and the service was horrible.",
            "Great place! Love the atmosphere and the pasta is fantastic.",
            "Average food, could be better. The staff was friendly though."
        ]
        
        self.expected_sentiments = ['positive', 'neutral', 'negative', 'positive', 'neutral']
    
    def test_analyze_single_review_positive(self):
        """Test analysis of a clearly positive review"""
        positive_review = "Amazing food! Excellent service! Highly recommend this place!"
        
        result = self.analyzer.analyze_sentiment(positive_review)
        
        # Check structure
        self.assertIn('sentiment', result)
        self.assertIn('confidence', result)
        self.assertIn('scores', result)
        
        # Check sentiment classification
        self.assertEqual(result['sentiment'], 'positive')
        
        # Check confidence is reasonable
        self.assertGreater(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Check scores structure
        self.assertIn('positive', result['scores'])
        self.assertIn('negative', result['scores'])
        self.assertIn('neutral', result['scores'])
    
    def test_analyze_single_review_negative(self):
        """Test analysis of a clearly negative review"""
        negative_review = "Terrible food! Awful service! Never going back!"
        
        result = self.analyzer.analyze_sentiment(negative_review)
        
        self.assertEqual(result['sentiment'], 'negative')
        self.assertGreater(result['confidence'], 0.0)
        self.assertGreater(result['scores']['negative'], result['scores']['positive'])
    
    def test_analyze_single_review_neutral(self):
        """Test analysis of a neutral review"""
        neutral_review = "The food was okay. Service was average. Nothing special."
        
        result = self.analyzer.analyze_sentiment(neutral_review)
        
        self.assertEqual(result['sentiment'], 'neutral')
        self.assertGreater(result['confidence'], 0.0)
    
    def test_analyze_reviews_batch(self):
        """Test batch analysis of multiple reviews"""
        reviews_df = pd.DataFrame({
            'review_id': range(1, len(self.test_reviews) + 1),
            'review_text': self.test_reviews,
            'restaurant_id': [1, 1, 2, 2, 3]
        })
        
        results = self.analyzer.analyze_reviews_batch(reviews_df)
        
        # Should return a list of results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.test_reviews))
        
        # Each result should have correct structure
        for i, result in enumerate(results):
            self.assertIn('review_id', result)
            self.assertIn('sentiment', result)
            self.assertIn('confidence', result)
            self.assertEqual(result['review_id'], i + 1)
    
    def test_get_restaurant_sentiment_score(self):
        """Test aggregation of sentiment scores for a restaurant"""
        # Create sample sentiment results
        sentiment_results = [
            {'review_id': 1, 'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.8},
            {'review_id': 2, 'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.9},
            {'review_id': 3, 'restaurant_id': 1, 'sentiment': 'negative', 'confidence': 0.7},
            {'review_id': 4, 'restaurant_id': 2, 'sentiment': 'neutral', 'confidence': 0.6}
        ]
        
        # Test for restaurant 1
        restaurant_sentiment = self.analyzer.get_restaurant_sentiment_score(
            sentiment_results, restaurant_id=1
        )
        
        # Check structure
        self.assertIn('overall_sentiment', restaurant_sentiment)
        self.assertIn('sentiment_score', restaurant_sentiment)
        self.assertIn('review_count', restaurant_sentiment)
        self.assertIn('sentiment_distribution', restaurant_sentiment)
        
        # Check values
        self.assertEqual(restaurant_sentiment['review_count'], 3)
        self.assertIn(restaurant_sentiment['overall_sentiment'], ['positive', 'negative', 'neutral'])
        self.assertIsInstance(restaurant_sentiment['sentiment_score'], (float, np.floating))
        
        # Check sentiment distribution
        dist = restaurant_sentiment['sentiment_distribution']
        self.assertIn('positive', dist)
        self.assertIn('negative', dist)
        self.assertIn('neutral', dist)
    
    def test_sentiment_consistency(self):
        """Test that sentiment analysis is consistent across multiple runs"""
        review = "This is a fantastic restaurant with amazing food!"
        
        result1 = self.analyzer.analyze_sentiment(review)
        result2 = self.analyzer.analyze_sentiment(review)
        
        # Should be identical
        self.assertEqual(result1['sentiment'], result2['sentiment'])
        self.assertAlmostEqual(result1['confidence'], result2['confidence'], places=5)
    
    def test_edge_case_empty_review(self):
        """Test handling of empty or whitespace-only reviews"""
        empty_reviews = ["", "   ", "\n\t", None]
        
        for review in empty_reviews:
            if review is not None:
                result = self.analyzer.analyze_sentiment(review)
                
                # Should handle gracefully
                self.assertIn('sentiment', result)
                self.assertIn('confidence', result)
                
                # Might be neutral or have low confidence
                self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_edge_case_very_short_review(self):
        """Test handling of very short reviews"""
        short_reviews = ["Good", "Bad", "OK", "Meh", "!"]
        
        for review in short_reviews:
            result = self.analyzer.analyze_sentiment(review)
            
            # Should handle gracefully
            self.assertIn('sentiment', result)
            self.assertIsInstance(result['confidence'], (float, np.floating))
    
    def test_edge_case_very_long_review(self):
        """Test handling of very long reviews"""
        long_review = "Great food! " * 100  # Very long review
        
        result = self.analyzer.analyze_sentiment(long_review)
        
        # Should handle gracefully
        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['confidence'], 0.0)
    
    def test_special_characters_handling(self):
        """Test handling of reviews with special characters"""
        special_reviews = [
            "Food was 👍👍👍 amazing!!! 😍",
            "Terrible... 😞😞😞 worst meal ever!!!",
            "Review with émojis and spëcial chars: ñice föod!",
            "ALL CAPS REVIEW WITH EXCLAMATION!!!!"
        ]
        
        for review in special_reviews:
            result = self.analyzer.analyze_sentiment(review)
            
            # Should handle gracefully
            self.assertIn('sentiment', result)
            self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_mixed_sentiment_review(self):
        """Test handling of reviews with mixed sentiments"""
        mixed_review = "The food was amazing and delicious, but the service was terrible and slow."
        
        result = self.analyzer.analyze_sentiment(mixed_review)
        
        # Should classify as one of the sentiments
        self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
        
        # Confidence might be lower for mixed sentiments
        self.assertGreater(result['confidence'], 0.0)
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing"""
        # Create larger batch of reviews
        large_reviews = pd.DataFrame({
            'review_id': range(1, 1001),
            'review_text': [f"This is test review number {i}" for i in range(1, 1001)],
            'restaurant_id': np.random.randint(1, 101, 1000)
        })
        
        # Time the batch processing
        result, execution_time = self.time_function(
            self.analyzer.analyze_reviews_batch, large_reviews
        )
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 30.0, 
                       f"Batch processing took {execution_time:.2f}s, expected < 30.0s")
        
        # Should return correct number of results
        self.assertEqual(len(result), 1000)
    
    def test_sentiment_score_calculation(self):
        """Test sentiment score calculation accuracy"""
        # Test with known sentiment distribution
        sentiment_results = [
            {'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.9},
            {'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.8},
            {'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.7},
            {'restaurant_id': 1, 'sentiment': 'negative', 'confidence': 0.6}
        ]
        
        restaurant_sentiment = self.analyzer.get_restaurant_sentiment_score(
            sentiment_results, restaurant_id=1
        )
        
        # Should be positive overall (3 positive vs 1 negative)
        self.assertEqual(restaurant_sentiment['overall_sentiment'], 'positive')
        
        # Sentiment score should reflect positive bias
        self.assertGreater(restaurant_sentiment['sentiment_score'], 0.0)
        
        # Distribution should be correct
        dist = restaurant_sentiment['sentiment_distribution']
        self.assertEqual(dist['positive'], 3)
        self.assertEqual(dist['negative'], 1)
        self.assertEqual(dist['neutral'], 0)
    
    def test_restaurant_no_reviews(self):
        """Test handling of restaurant with no reviews"""
        empty_results = []
        
        restaurant_sentiment = self.analyzer.get_restaurant_sentiment_score(
            empty_results, restaurant_id=999
        )
        
        # Should handle gracefully
        self.assertEqual(restaurant_sentiment['review_count'], 0)
        self.assertEqual(restaurant_sentiment['overall_sentiment'], 'neutral')
        self.assertEqual(restaurant_sentiment['sentiment_score'], 0.0)
    
    def test_confidence_thresholds(self):
        """Test that confidence thresholds work appropriately"""
        # Test with low confidence results
        low_confidence_results = [
            {'restaurant_id': 1, 'sentiment': 'positive', 'confidence': 0.1},
            {'restaurant_id': 1, 'sentiment': 'negative', 'confidence': 0.2}
        ]
        
        restaurant_sentiment = self.analyzer.get_restaurant_sentiment_score(
            low_confidence_results, restaurant_id=1
        )
        
        # Should still process but might affect overall score
        self.assertIsInstance(restaurant_sentiment['sentiment_score'], (float, np.floating))
        self.assertEqual(restaurant_sentiment['review_count'], 2)
    
    @slow_test
    def test_stress_testing(self):
        """Stress test with various edge cases"""
        stress_reviews = [
            "",  # Empty
            "a",  # Single character
            "A" * 10000,  # Very long
            "🎉🎊🎈" * 50,  # Only emojis
            "123456789",  # Only numbers
            "!@#$%^&*()",  # Only special chars
            "Review in multiple languages: Muy bueno! Très bien! Molto bene!",
            "Mixed case AND symbols!!! 😀😀😀"
        ]
        
        for review in stress_reviews:
            try:
                result = self.analyzer.analyze_sentiment(review)
                self.assertIn('sentiment', result)
                self.assertIn('confidence', result)
            except Exception as e:
                self.fail(f"Failed on review '{review[:50]}...': {str(e)}")

if __name__ == '__main__':
    unittest.main()
