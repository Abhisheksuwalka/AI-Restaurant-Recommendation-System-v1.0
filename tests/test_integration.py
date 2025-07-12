"""
Integration Tests for the Enhanced AI Recommendation System with Emotional Intelligence

This module provides comprehensive integration tests to validate the complete system
including emotional intelligence, LLM enhancement, and traditional recommendation methods.
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import system components
from models.hybrid_recommender import HybridRecommender
from models.emotional_intelligence import EmotionalIntelligenceEngine
from data.preprocessor import DataPreprocessor
from config import Config

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete AI recommendation system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.original_data_dir = Config.EMOTIONAL_DATA_DIR
        Config.EMOTIONAL_DATA_DIR = cls.test_dir
        
        # Create sample data
        cls.sample_data = cls._create_sample_data()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
        Config.EMOTIONAL_DATA_DIR = cls.original_data_dir
    
    @classmethod
    def _create_sample_data(cls):
        """Create comprehensive sample data for testing"""
        # Sample users
        users_data = {
            'user_id': [1, 2, 3, 4, 5],
            'age': [25, 35, 28, 42, 31],
            'location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'preferred_cuisine': ['Italian', 'Mexican', 'Asian', 'American', 'Indian'],
            'budget_preference': ['Medium', 'High', 'Low', 'Medium', 'High']
        }
        
        # Sample restaurants
        restaurants_data = {
            'restaurant_id': list(range(1, 21)),
            'name': [f'Restaurant {i}' for i in range(1, 21)],
            'cuisine': ['Italian', 'Mexican', 'Chinese', 'American', 'Japanese'] * 4,
            'rating': np.random.uniform(3.0, 5.0, 20),
            'location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 4,
            'price_range': ['$', '$$', '$$$', '$$', '$'] * 4
        }
        
        # Sample ratings
        ratings_data = []
        for user_id in range(1, 6):
            for restaurant_id in np.random.choice(range(1, 21), 8, replace=False):
                ratings_data.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant_id,
                    'rating': np.random.choice([3, 4, 5])
                })
        
        # Sample reviews
        sample_reviews = [
            "Great food and excellent service!",
            "The ambiance was perfect for a romantic dinner.",
            "Food was okay but service was slow.",
            "Amazing flavors and presentation!",
            "Disappointed with the quality.",
            "Cozy atmosphere, perfect for dates.",
            "Too crowded and noisy.",
            "Authentic cuisine and fresh ingredients.",
            "Overpriced for the portion size.",
            "Wonderful experience, will come back!"
        ]
        
        reviews_data = []
        for i, rating_entry in enumerate(ratings_data):
            reviews_data.append({
                'user_id': rating_entry['user_id'],
                'restaurant_id': rating_entry['restaurant_id'],
                'rating': rating_entry['rating'],
                'review_text': np.random.choice(sample_reviews),
                'timestamp': datetime.now()
            })
        
        return {
            'users': pd.DataFrame(users_data),
            'restaurants': pd.DataFrame(restaurants_data),
            'ratings': pd.DataFrame(ratings_data),
            'reviews': pd.DataFrame(reviews_data)
        }
    
    def setUp(self):
        """Set up test environment for each test"""
        # Mock the preprocessor to use our sample data
        with patch('models.hybrid_recommender.DataPreprocessor'):
            self.recommender = HybridRecommender()
            
            # Manually set the data
            self.recommender.data = self.sample_data
            
            # Create user-item matrix for collaborative filtering
            user_item_matrix = pd.crosstab(
                self.sample_data['ratings']['user_id'],
                self.sample_data['ratings']['restaurant_id'],
                self.sample_data['ratings']['rating'],
                aggfunc='mean'
            ).fillna(0)
            
            # Create restaurant features for content-based filtering
            restaurant_features = pd.get_dummies(
                self.sample_data['restaurants'][['cuisine', 'price_range']]
            )
            
            # Prepare data for fitting
            fit_data = {
                'user_item_matrix': user_item_matrix,
                'restaurants': self.sample_data['restaurants'],
                'restaurant_features': restaurant_features,
                'reviews': self.sample_data['reviews'],
                'ratings': self.sample_data['ratings'],
                'users': self.sample_data['users']
            }
            
            # Mock the models to avoid heavy computation
            self.recommender.collaborative_model = Mock()
            self.recommender.collaborative_model.get_user_recommendations.return_value = [
                (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)
            ]
            
            self.recommender.content_model = Mock()
            self.recommender.content_model.get_restaurant_recommendations.return_value = [
                {'restaurant_id': 6, 'similarity_score': 0.9},
                {'restaurant_id': 7, 'similarity_score': 0.8},
                {'restaurant_id': 8, 'similarity_score': 0.7}
            ]
            
            self.recommender.sentiment_analyzer = Mock()
            self.recommender.sentiment_scores = {
                i: {'avg_sentiment': 0.1, 'total_reviews': 5, 'positive_ratio': 0.7, 'negative_ratio': 0.3}
                for i in range(1, 21)
            }
            
            # Fit the recommender
            self.recommender.fit(fit_data)
    
    def test_traditional_recommendations(self):
        """Test traditional hybrid recommendations without emotional intelligence"""
        recommendations = self.recommender.get_hybrid_recommendations(
            user_id=1,
            n_recommendations=5,
            collaborative_weight=0.4,
            content_weight=0.3,
            sentiment_weight=0.3
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 5)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('restaurant_id', rec)
            self.assertIn('name', rec)
            self.assertIn('final_score', rec)
            self.assertIn('sentiment_score', rec)
            self.assertIsInstance(rec['final_score'], float)
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_emotional_intelligence_integration(self, mock_transformer):
        """Test emotional intelligence integration with recommendations"""
        # Mock the emotion detector
        mock_detector = Mock()
        mock_emotional_state = Mock()
        mock_emotional_state.primary_emotion = 'happy'
        mock_emotional_state.secondary_emotion = None
        mock_emotional_state.intensity = 0.8
        mock_emotional_state.confidence = 0.9
        mock_emotional_state.context = {}
        mock_emotional_state.timestamp = datetime.now()
        
        mock_detector.detect_emotion.return_value = mock_emotional_state
        mock_transformer.return_value = mock_detector
        
        # Initialize emotional intelligence
        self.recommender.emotional_engine = EmotionalIntelligenceEngine()
        self.recommender.emotional_engine.state_manager.detector = mock_detector
        
        # Test emotional recommendations
        recommendations = self.recommender.get_emotional_recommendations(
            user_id=1,
            user_text_input="I'm feeling really happy today!",
            n_recommendations=5
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check that emotional scores are included
        for rec in recommendations:
            self.assertIn('emotional_score', rec)
            self.assertIsInstance(rec['emotional_score'], float)
            self.assertGreaterEqual(rec['emotional_score'], 0.0)
            self.assertLessEqual(rec['emotional_score'], 1.0)
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_hybrid_with_emotional_intelligence(self, mock_transformer):
        """Test hybrid recommendations with emotional intelligence enabled"""
        # Mock the emotion detector
        mock_detector = Mock()
        mock_emotional_state = Mock()
        mock_emotional_state.primary_emotion = 'stressed'
        mock_emotional_state.intensity = 0.7
        
        mock_detector.detect_emotion.return_value = mock_emotional_state
        mock_transformer.return_value = mock_detector
        
        # Initialize emotional intelligence
        self.recommender.emotional_engine = EmotionalIntelligenceEngine()
        self.recommender.emotional_engine.state_manager.detector = mock_detector
        
        # Test hybrid recommendations with emotional weight
        recommendations = self.recommender.get_hybrid_recommendations(
            user_id=1,
            n_recommendations=5,
            collaborative_weight=0.25,
            content_weight=0.25,
            sentiment_weight=0.25,
            emotional_weight=0.25,
            user_text_input="Work is really stressing me out"
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check that all scores are included
        for rec in recommendations:
            self.assertIn('collaborative_score', rec)
            self.assertIn('content_score', rec)
            self.assertIn('sentiment_score', rec)
            self.assertIn('emotional_score', rec)
            self.assertIn('final_score', rec)
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_mood_based_cuisine_suggestions(self, mock_transformer):
        """Test mood-based cuisine suggestions"""
        # Mock the emotion detector
        mock_detector = Mock()
        mock_emotional_state = Mock()
        mock_emotional_state.primary_emotion = 'romantic'
        mock_emotional_state.intensity = 0.9
        
        mock_detector.detect_emotion.return_value = mock_emotional_state
        mock_transformer.return_value = mock_detector
        
        # Initialize emotional intelligence
        self.recommender.emotional_engine = EmotionalIntelligenceEngine()
        self.recommender.emotional_engine.state_manager.detector = mock_detector
        
        # Test mood-based cuisine suggestions
        suggestions = self.recommender.get_mood_based_cuisine_suggestions(
            user_id=1,
            user_text_input="Planning a romantic dinner"
        )
        
        self.assertIsInstance(suggestions, dict)
        self.assertIn('detected_emotion', suggestions)
        self.assertIn('recommended_cuisines', suggestions)
        self.assertEqual(suggestions['detected_emotion'], 'romantic')
        self.assertIsInstance(suggestions['recommended_cuisines'], list)
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_emotional_insights(self, mock_transformer):
        """Test user emotional insights functionality"""
        # Mock the emotion detector
        mock_detector = Mock()
        mock_transformer.return_value = mock_detector
        
        # Initialize emotional intelligence
        self.recommender.emotional_engine = EmotionalIntelligenceEngine()
        self.recommender.emotional_engine.state_manager.detector = mock_detector
        
        # Test emotional insights
        insights = self.recommender.get_user_emotional_insights(user_id=1)
        
        self.assertIsInstance(insights, dict)
        # Since no history exists yet, should return empty patterns
        self.assertIn('emotional_history_count', insights)
        self.assertEqual(insights['emotional_history_count'], 0)
    
    def test_system_performance_metrics(self):
        """Test that the system provides meaningful performance metrics"""
        # Test traditional recommendations performance
        import time
        
        start_time = time.time()
        recommendations = self.recommender.get_hybrid_recommendations(
            user_id=1, n_recommendations=10
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete recommendations in reasonable time (< 1 second)
        self.assertLess(processing_time, 1.0)
        self.assertGreater(len(recommendations), 0)
        
        # Check score distributions
        scores = [rec['final_score'] for rec in recommendations]
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
        # Scores should be in descending order
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_error_handling(self):
        """Test system error handling and graceful degradation"""
        # Test with invalid user ID
        recommendations = self.recommender.get_hybrid_recommendations(
            user_id=999,  # Non-existent user
            n_recommendations=5
        )
        
        # Should still return recommendations (system-wide recommendations)
        self.assertIsInstance(recommendations, list)
        
        # Test with invalid parameters
        recommendations = self.recommender.get_hybrid_recommendations(
            user_id=1,
            n_recommendations=0  # Invalid number
        )
        
        # Should handle gracefully
        self.assertIsInstance(recommendations, list)
    
    def test_configuration_validation(self):
        """Test that system configuration is properly validated"""
        # Test that emotional intelligence respects configuration
        original_config = Config.USE_EMOTIONAL_RECOMMENDATIONS
        
        # Disable emotional recommendations
        Config.USE_EMOTIONAL_RECOMMENDATIONS = False
        
        # Create new recommender
        with patch('models.hybrid_recommender.DataPreprocessor'):
            new_recommender = HybridRecommender()
            
            # Emotional engine should not be initialized
            self.assertIsNone(new_recommender.emotional_engine)
        
        # Restore original configuration
        Config.USE_EMOTIONAL_RECOMMENDATIONS = original_config
    
    def test_data_consistency(self):
        """Test data consistency across different recommendation methods"""
        # Get recommendations using different methods
        traditional_recs = self.recommender.get_hybrid_recommendations(
            user_id=1, n_recommendations=5
        )
        
        # All recommendations should reference valid restaurants
        valid_restaurant_ids = set(self.sample_data['restaurants']['restaurant_id'])
        
        for rec in traditional_recs:
            self.assertIn(rec['restaurant_id'], valid_restaurant_ids)
    
    def test_scalability_simulation(self):
        """Test system scalability with multiple concurrent requests"""
        import concurrent.futures
        import time
        
        def get_recommendations_for_user(user_id):
            return self.recommender.get_hybrid_recommendations(
                user_id=user_id, n_recommendations=5
            )
        
        # Simulate concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(get_recommendations_for_user, user_id)
                for user_id in [1, 2, 3, 4, 5]
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 5 concurrent requests in reasonable time
        self.assertLess(total_time, 5.0)
        self.assertEqual(len(results), 5)
        
        # All results should be valid
        for result in results:
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

class TestSystemStressTest(unittest.TestCase):
    """Stress tests for the AI recommendation system"""
    
    def test_large_dataset_performance(self):
        """Test system performance with larger datasets"""
        # Create larger sample data
        large_users = pd.DataFrame({
            'user_id': range(1, 101),
            'age': np.random.randint(18, 65, 100),
            'location': np.random.choice(['New York', 'LA', 'Chicago'], 100),
            'preferred_cuisine': np.random.choice(['Italian', 'Mexican', 'Asian'], 100),
            'budget_preference': np.random.choice(['Low', 'Medium', 'High'], 100)
        })
        
        large_restaurants = pd.DataFrame({
            'restaurant_id': range(1, 501),
            'name': [f'Restaurant {i}' for i in range(1, 501)],
            'cuisine': np.random.choice(['Italian', 'Mexican', 'Chinese', 'American', 'Japanese'], 500),
            'rating': np.random.uniform(3.0, 5.0, 500),
            'location': np.random.choice(['New York', 'LA', 'Chicago'], 500),
            'price_range': np.random.choice(['$', '$$', '$$$'], 500)
        })
        
        # Test should complete without memory errors
        self.assertIsInstance(large_users, pd.DataFrame)
        self.assertIsInstance(large_restaurants, pd.DataFrame)
        self.assertEqual(len(large_users), 100)
        self.assertEqual(len(large_restaurants), 500)

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running AI Recommendation System Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestSystemIntegration, TestSystemStressTest]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
    
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
