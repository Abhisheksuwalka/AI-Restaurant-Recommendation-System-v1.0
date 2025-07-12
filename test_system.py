import unittest
import pandas as pd
from data.preprocessor import DataPreprocessor
from models.hybrid_recommender import HybridRecommender

class TestRecommendationSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_data()
        self.preprocessor.preprocess_restaurants()
        self.data = self.preprocessor.get_processed_data()
        
        self.recommender = HybridRecommender()
        self.recommender.fit(self.data)
    
    def test_data_loading(self):
        """Test if data loads correctly"""
        self.assertGreater(len(self.data['restaurants']), 0)
        self.assertGreater(len(self.data['users']), 0)
        self.assertGreater(len(self.data['ratings']), 0)
    
    def test_recommendations(self):
        """Test if recommendations are generated"""
        user_id = self.data['users']['user_id'].iloc[0]
        recommendations = self.recommender.get_hybrid_recommendations(user_id, 5)
        
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 5)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        self.assertIsNotNone(self.recommender.sentiment_results)
        self.assertGreater(len(self.recommender.sentiment_results), 0)

if __name__ == "__main__":
    unittest.main()
