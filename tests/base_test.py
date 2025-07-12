"""
Base Test Classes for AI Recommendation System

This module provides base test classes with common setup, teardown,
and utility methods for all test modules in the system.
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from typing import Dict, Any, Optional, List
import json
import logging

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tests.test_factory import TestDataFactory, create_test_config
from config import Config

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures"""
        cls.test_config = create_test_config()
        cls.factory = TestDataFactory()
        cls.temp_dir = tempfile.mkdtemp()
        
        # Override config for testing
        cls.original_config = {}
        for key, value in cls.test_config.items():
            if hasattr(Config, key):
                cls.original_config[key] = getattr(Config, key)
                setattr(Config, key, value)
        
        # Set up logging for tests
        logging.basicConfig(level=logging.WARNING)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures"""
        # Restore original config
        for key, value in cls.original_config.items():
            setattr(Config, key, value)
        
        # Clean up temp directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up test fixtures for each test"""
        self.test_data = self.create_test_data()
        
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    def create_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create standard test data for each test"""
        users = self.factory.create_users(50)
        restaurants = self.factory.create_restaurants(100)
        ratings = self.factory.create_ratings(users, restaurants, 500)
        reviews = self.factory.create_reviews(ratings)
        
        return {
            'users': users,
            'restaurants': restaurants,
            'ratings': ratings,
            'reviews': reviews
        }
    
    def assertDataFrameEqual(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                           msg: Optional[str] = None):
        """Assert that two DataFrames are equal"""
        try:
            pd.testing.assert_frame_equal(df1, df2)
        except AssertionError as e:
            if msg:
                raise AssertionError(f"{msg}: {str(e)}")
            else:
                raise
    
    def assertArrayAlmostEqual(self, arr1: np.ndarray, arr2: np.ndarray, 
                              places: int = 7, msg: Optional[str] = None):
        """Assert that two numpy arrays are almost equal"""
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=places)
    
    def assertRecommendationsValid(self, recommendations: List[Dict], 
                                  n_expected: int, required_fields: List[str]):
        """Assert that recommendations have valid structure"""
        self.assertEqual(len(recommendations), n_expected, 
                        f"Expected {n_expected} recommendations, got {len(recommendations)}")
        
        for i, rec in enumerate(recommendations):
            for field in required_fields:
                self.assertIn(field, rec, 
                            f"Recommendation {i} missing required field '{field}'")
            
            # Check data types
            if 'restaurant_id' in rec:
                self.assertIsInstance(rec['restaurant_id'], (int, np.integer))
            if 'score' in rec:
                self.assertIsInstance(rec['score'], (float, np.floating))
                self.assertGreaterEqual(rec['score'], 0.0)
                self.assertLessEqual(rec['score'], 1.0)

class ModelTestCase(BaseTestCase):
    """Base test case for model testing"""
    
    def setUp(self):
        super().setUp()
        self.user_item_matrix = self.create_user_item_matrix()
        self.restaurant_features = self.create_restaurant_features()
    
    def create_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item matrix from test ratings"""
        return self.test_data['ratings'].pivot_table(
            index='user_id',
            columns='restaurant_id',
            values='rating',
            fill_value=0
        )
    
    def create_restaurant_features(self) -> np.ndarray:
        """Create feature matrix for restaurants"""
        restaurants = self.test_data['restaurants']
        # Simple feature encoding for testing
        features = restaurants[['rating', 'num_reviews']].values
        return (features - features.mean(axis=0)) / features.std(axis=0)
    
    def assertModelTrained(self, model):
        """Assert that a model has been properly trained"""
        self.assertTrue(hasattr(model, 'fit'), "Model should have fit method")
        # Add model-specific assertions in subclasses

class IntegrationTestCase(BaseTestCase):
    """Base test case for integration testing"""
    
    def setUp(self):
        super().setUp()
        self.integration_data = self.create_integration_data()
    
    def create_integration_data(self) -> Dict:
        """Create comprehensive data for integration tests"""
        data = self.create_test_data()
        
        # Add processed data
        data['user_item_matrix'] = data['ratings'].pivot_table(
            index='user_id',
            columns='restaurant_id', 
            values='rating',
            fill_value=0
        )
        
        # Add restaurant features
        restaurants = data['restaurants']
        feature_cols = ['rating', 'num_reviews']
        data['restaurant_features'] = restaurants[feature_cols].values
        
        return data
    
    def assertSystemHealthy(self, system):
        """Assert that the system is in a healthy state"""
        self.assertTrue(hasattr(system, 'fit'), "System should have fit method")
        self.assertTrue(hasattr(system, 'get_hybrid_recommendations'), 
                       "System should have recommendation method")

class PerformanceTestCase(BaseTestCase):
    """Base test case for performance testing"""
    
    def setUp(self):
        super().setUp()
        self.performance_data = self.factory.create_performance_test_data(scale_factor=1)
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def assertExecutionTime(self, func, max_time: float, *args, **kwargs):
        """Assert that function executes within time limit"""
        result, execution_time = self.time_function(func, *args, **kwargs)
        self.assertLessEqual(execution_time, max_time,
                           f"Function took {execution_time:.2f}s, expected < {max_time}s")
        return result

class MockTestCase(BaseTestCase):
    """Base test case with mocking utilities"""
    
    def create_mock_llm_enhancer(self):
        """Create a mock LLM enhancer for testing"""
        mock_enhancer = Mock()
        mock_enhancer.generate_explanation.return_value = "Test explanation"
        mock_enhancer.get_cuisine_suggestions.return_value = ["Italian", "Mexican"]
        mock_enhancer.is_available.return_value = True
        return mock_enhancer
    
    def create_mock_emotional_engine(self):
        """Create a mock emotional intelligence engine"""
        mock_engine = Mock()
        mock_engine.detect_emotion.return_value = {
            'primary_emotion': 'happy',
            'confidence': 0.8,
            'all_emotions': {'happy': 0.8, 'excited': 0.2}
        }
        mock_engine.get_emotional_restaurant_scores.return_value = {
            1: 0.9, 2: 0.7, 3: 0.8
        }
        return mock_engine
    
    def create_mock_api_response(self, status_code: int = 200, 
                                data: Optional[Dict] = None):
        """Create a mock API response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data or {}
        return mock_response

# Test decorators
def skip_if_no_llm(test_func):
    """Skip test if LLM enhancement is not available"""
    def wrapper(self):
        if not Config.USE_LLM_ENHANCEMENT:
            self.skipTest("LLM enhancement not available")
        return test_func(self)
    return wrapper

def skip_if_no_emotional_ai(test_func):
    """Skip test if emotional AI is not available"""
    def wrapper(self):
        if not Config.USE_EMOTIONAL_RECOMMENDATIONS:
            self.skipTest("Emotional AI not available")
        return test_func(self)
    return wrapper

def slow_test(test_func):
    """Mark test as slow"""
    test_func._slow_test = True
    return test_func

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_temporary_file(content: str, suffix: str = '.json') -> str:
        """Create a temporary file with content"""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
        except:
            os.close(fd)
            raise
        return path
    
    @staticmethod
    def assert_performance_acceptable(execution_time: float, 
                                    max_time: float, 
                                    operation_name: str):
        """Assert that performance is acceptable"""
        if execution_time > max_time:
            raise AssertionError(
                f"{operation_name} took {execution_time:.2f}s, "
                f"which exceeds the maximum allowed time of {max_time}s"
            )
    
    @staticmethod
    def generate_user_input_variations() -> List[str]:
        """Generate various user input scenarios for testing"""
        return [
            "I'm feeling stressed and need comfort food",
            "Excited about my date tonight!",
            "Just want something quick and healthy",
            "Feeling adventurous, surprise me!",
            "Family dinner, need kid-friendly options",
            "Celebrating a promotion",
            "Rainy day, want something cozy",
            "Hot summer day, need something refreshing"
        ]
