"""
Comprehensive Tests for Data Preprocessor

Tests all aspects of data preprocessing including loading, transforming,
encoding, and creating feature matrices for the recommendation system.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from tests.base_test import BaseTestCase, TestUtils
from data.preprocessor import DataPreprocessor

class TestDataPreprocessor(BaseTestCase):
    """Test the DataPreprocessor class functionality"""
    
    def setUp(self):
        super().setUp()
        self.preprocessor = DataPreprocessor()
        
        # Create sample CSV files for testing
        self.temp_files = self._create_temp_csv_files()
    
    def tearDown(self):
        super().tearDown()
        # Clean up temporary files
        for file_path in self.temp_files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def _create_temp_csv_files(self):
        """Create temporary CSV files for testing"""
        temp_files = {}
        
        # Save test data to temporary CSV files
        temp_files['restaurants'] = TestUtils.create_temporary_file(
            self.test_data['restaurants'].to_csv(index=False), 
            suffix='.csv'
        )
        temp_files['users'] = TestUtils.create_temporary_file(
            self.test_data['users'].to_csv(index=False), 
            suffix='.csv'
        )
        temp_files['ratings'] = TestUtils.create_temporary_file(
            self.test_data['ratings'].to_csv(index=False), 
            suffix='.csv'
        )
        temp_files['reviews'] = TestUtils.create_temporary_file(
            self.test_data['reviews'].to_csv(index=False), 
            suffix='.csv'
        )
        
        return temp_files
    
    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading"""
        # Mock pandas.read_csv to return our test data
        mock_read_csv.side_effect = [
            self.test_data['restaurants'],
            self.test_data['users'], 
            self.test_data['ratings'],
            self.test_data['reviews']
        ]
        
        self.preprocessor.load_data()
        
        self.assertEqual(len(self.preprocessor.restaurants), len(self.test_data['restaurants']))
        self.assertEqual(len(self.preprocessor.users), len(self.test_data['users']))
        self.assertEqual(len(self.preprocessor.ratings), len(self.test_data['ratings']))
        self.assertEqual(len(self.preprocessor.reviews), len(self.test_data['reviews']))
        
        # Verify correct columns are present
        expected_restaurant_cols = ['restaurant_id', 'name', 'cuisine', 'location', 'price_range', 'rating']
        for col in expected_restaurant_cols:
            self.assertIn(col, self.preprocessor.restaurants.columns)
    
    @patch('pandas.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test handling of missing data files"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data()
    
    def test_preprocess_restaurants_encoding(self):
        """Test restaurant data preprocessing and encoding"""
        self.preprocessor.restaurants = self.test_data['restaurants'].copy()
        
        processed = self.preprocessor.preprocess_restaurants()
        
        # Check if categorical variables are encoded
        categorical_cols = ['cuisine', 'location', 'price_range']
        for col in categorical_cols:
            encoded_col = f'{col}_encoded'
            self.assertIn(encoded_col, processed.columns)
            
            # Check if encoding is valid (integers)
            self.assertTrue(processed[encoded_col].dtype in [np.int32, np.int64])
            
            # Check if label encoder is stored
            self.assertIn(col, self.preprocessor.label_encoders)
    
    def test_preprocess_restaurants_feature_matrix(self):
        """Test creation of restaurant feature matrix"""
        self.preprocessor.restaurants = self.test_data['restaurants'].copy()
        
        self.preprocessor.preprocess_restaurants()
        
        # Check if feature matrix is created
        self.assertIsNotNone(self.preprocessor.restaurant_features)
        
        # Check dimensions
        n_restaurants = len(self.test_data['restaurants'])
        expected_features = 4  # cuisine_encoded, location_encoded, price_range_encoded, rating
        self.assertEqual(self.preprocessor.restaurant_features.shape, (n_restaurants, expected_features))
        
        # Check if features are normalized (mean ~0, std ~1)
        means = np.mean(self.preprocessor.restaurant_features, axis=0)
        stds = np.std(self.preprocessor.restaurant_features, axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros_like(means), decimal=1)
        np.testing.assert_array_almost_equal(stds, np.ones_like(stds), decimal=1)
    
    def test_create_user_item_matrix_structure(self):
        """Test user-item matrix creation and structure"""
        self.preprocessor.ratings = self.test_data['ratings'].copy()
        
        user_item_matrix = self.preprocessor.create_user_item_matrix()
        
        # Check if it's a proper pivot table
        self.assertIsInstance(user_item_matrix, pd.DataFrame)
        
        # Check dimensions
        unique_users = self.test_data['ratings']['user_id'].nunique()
        unique_restaurants = self.test_data['ratings']['restaurant_id'].nunique()
        self.assertEqual(user_item_matrix.shape[0], unique_users)
        self.assertEqual(user_item_matrix.shape[1], unique_restaurants)
        
        # Check if missing values are filled with 0
        self.assertTrue((user_item_matrix.isna().sum().sum() == 0))
    
    def test_create_user_item_matrix_values(self):
        """Test user-item matrix values are correct"""
        # Create specific test data to verify exact values
        test_ratings = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'restaurant_id': [1, 2, 1, 3, 2], 
            'rating': [5.0, 4.0, 3.0, 4.5, 2.0]
        })
        
        self.preprocessor.ratings = test_ratings
        user_item_matrix = self.preprocessor.create_user_item_matrix()
        
        # Check specific values
        self.assertEqual(user_item_matrix.loc[1, 1], 5.0)
        self.assertEqual(user_item_matrix.loc[1, 2], 4.0)
        self.assertEqual(user_item_matrix.loc[2, 1], 3.0)
        self.assertEqual(user_item_matrix.loc[2, 3], 4.5)
        self.assertEqual(user_item_matrix.loc[3, 2], 2.0)
        
        # Check zero values for non-rated items
        self.assertEqual(user_item_matrix.loc[1, 3], 0.0)
        self.assertEqual(user_item_matrix.loc[3, 1], 0.0)
    
    def test_get_processed_data_completeness(self):
        """Test that get_processed_data returns complete data structure"""
        # Set up preprocessor with all data
        self.preprocessor.restaurants = self.test_data['restaurants'].copy()
        self.preprocessor.users = self.test_data['users'].copy()
        self.preprocessor.ratings = self.test_data['ratings'].copy()
        self.preprocessor.reviews = self.test_data['reviews'].copy()
        
        # Preprocess data
        self.preprocessor.preprocess_restaurants()
        
        # Get processed data
        processed_data = self.preprocessor.get_processed_data()
        
        # Check all required keys are present
        required_keys = [
            'restaurants', 'users', 'ratings', 'reviews',
            'user_item_matrix', 'restaurant_features', 'label_encoders'
        ]
        
        for key in required_keys:
            self.assertIn(key, processed_data, f"Missing key: {key}")
        
        # Check data types
        self.assertIsInstance(processed_data['restaurants'], pd.DataFrame)
        self.assertIsInstance(processed_data['users'], pd.DataFrame)
        self.assertIsInstance(processed_data['ratings'], pd.DataFrame)
        self.assertIsInstance(processed_data['reviews'], pd.DataFrame)
        self.assertIsInstance(processed_data['user_item_matrix'], pd.DataFrame)
        self.assertIsInstance(processed_data['restaurant_features'], np.ndarray)
        self.assertIsInstance(processed_data['label_encoders'], dict)
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent across multiple runs"""
        # Set up data
        self.preprocessor.restaurants = self.test_data['restaurants'].copy()
        
        # First preprocessing
        processed1 = self.preprocessor.preprocess_restaurants()
        features1 = self.preprocessor.restaurant_features.copy()
        encoders1 = {k: v.classes_.copy() for k, v in self.preprocessor.label_encoders.items()}
        
        # Reset and do second preprocessing
        preprocessor2 = DataPreprocessor()
        preprocessor2.restaurants = self.test_data['restaurants'].copy()
        processed2 = preprocessor2.preprocess_restaurants()
        features2 = preprocessor2.restaurant_features.copy()
        encoders2 = {k: v.classes_.copy() for k, v in preprocessor2.label_encoders.items()}
        
        # Check consistency
        pd.testing.assert_frame_equal(processed1, processed2)
        np.testing.assert_array_equal(features1, features2)
        
        # Check encoder consistency
        for key in encoders1:
            np.testing.assert_array_equal(encoders1[key], encoders2[key])
    
    def test_edge_cases_empty_data(self):
        """Test handling of edge cases like empty data"""
        # Test with empty DataFrames
        empty_restaurants = pd.DataFrame(columns=['restaurant_id', 'cuisine', 'location', 'price_range', 'rating'])
        self.preprocessor.restaurants = empty_restaurants
        
        # Should not crash but return empty results
        processed = self.preprocessor.preprocess_restaurants()
        self.assertEqual(len(processed), 0)
        self.assertEqual(self.preprocessor.restaurant_features.shape[0], 0)
    
    def test_edge_cases_single_category(self):
        """Test handling of single category in categorical variables"""
        # Create data with single category
        single_cuisine = self.test_data['restaurants'].copy()
        single_cuisine['cuisine'] = 'Italian'  # All same cuisine
        
        self.preprocessor.restaurants = single_cuisine
        processed = self.preprocessor.preprocess_restaurants()
        
        # Should handle gracefully
        self.assertIn('cuisine_encoded', processed.columns)
        # All values should be the same (encoded as 0)
        self.assertTrue((processed['cuisine_encoded'] == 0).all())
    
    def test_performance_large_dataset(self):
        """Test performance with larger datasets"""
        # Create larger test data
        large_data = self.factory.create_performance_test_data(scale_factor=1)
        
        self.preprocessor.restaurants = large_data['restaurants']
        self.preprocessor.ratings = large_data['ratings']
        
        # Time the preprocessing
        result, execution_time = self.time_function(
            self.preprocessor.preprocess_restaurants
        )
        
        # Should complete within reasonable time (5 seconds for test data)
        self.assertLess(execution_time, 5.0, 
                       f"Preprocessing took {execution_time:.2f}s, expected < 5.0s")
        
        # Time user-item matrix creation
        result, execution_time = self.time_function(
            self.preprocessor.create_user_item_matrix
        )
        
        self.assertLess(execution_time, 3.0,
                       f"User-item matrix creation took {execution_time:.2f}s, expected < 3.0s")

if __name__ == '__main__':
    unittest.main()
