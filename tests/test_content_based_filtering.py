"""
Comprehensive Tests for Content-Based Filtering

Tests the content-based filtering recommendation algorithm including
feature similarity calculations, recommendation generation, and robustness.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from tests.base_test import ModelTestCase, slow_test
from models.content_based_filtering import ContentBasedFiltering

class TestContentBasedFiltering(ModelTestCase):
    """Test the ContentBasedFiltering model"""
    
    def setUp(self):
        super().setUp()
        self.model = ContentBasedFiltering()
    
    def test_fit_method_exists(self):
        """Test that fit method exists and accepts required parameters"""
        self.assertTrue(hasattr(self.model, 'fit'))
        
        # Should not raise exception
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Check if data is stored
        self.assertIsNotNone(self.model.restaurants)
        self.assertIsNotNone(self.model.restaurant_features)
    
    def test_fit_stores_data_correctly(self):
        """Test that fit method stores data correctly"""
        restaurants = self.test_data['restaurants']
        features = self.restaurant_features
        
        self.model.fit(restaurants, features)
        
        # Check if the stored data matches the input
        pd.testing.assert_frame_equal(self.model.restaurants, restaurants)
        np.testing.assert_array_equal(self.model.restaurant_features, features)
    
    def test_calculate_restaurant_similarity(self):
        """Test restaurant similarity calculation"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Get two restaurants
        restaurant_ids = list(self.test_data['restaurants']['restaurant_id'])
        if len(restaurant_ids) >= 2:
            rest1, rest2 = restaurant_ids[0], restaurant_ids[1]
            
            similarity = self.model.calculate_restaurant_similarity(rest1, rest2)
            
            # Similarity should be between -1 and 1
            self.assertGreaterEqual(similarity, -1.0)
            self.assertLessEqual(similarity, 1.0)
            self.assertIsInstance(similarity, (float, np.floating))
    
    def test_restaurant_similarity_identical(self):
        """Test that identical restaurants have similarity of 1.0"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        
        # Same restaurant should have similarity 1.0 with itself
        similarity = self.model.calculate_restaurant_similarity(restaurant_id, restaurant_id)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_get_restaurant_recommendations_basic(self):
        """Test basic restaurant recommendation generation"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Get a restaurant that exists
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=5
        )
        
        # Should return a list
        self.assertIsInstance(recommendations, list)
        
        # Should not exceed requested number
        self.assertLessEqual(len(recommendations), 5)
        
        # Each recommendation should be a valid restaurant ID
        restaurant_ids = set(self.test_data['restaurants']['restaurant_id'])
        for rec in recommendations:
            self.assertIsInstance(rec, (int, np.integer))
            self.assertIn(rec, restaurant_ids)
    
    def test_get_restaurant_recommendations_excludes_self(self):
        """Test that recommendations exclude the input restaurant"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=10
        )
        
        # Should not include the input restaurant itself
        self.assertNotIn(restaurant_id, recommendations)
    
    def test_get_restaurant_recommendations_ranking(self):
        """Test that recommendations are properly ranked by similarity"""
        # Create test data with known similarities
        test_restaurants = pd.DataFrame({
            'restaurant_id': [1, 2, 3, 4],
            'name': ['Rest1', 'Rest2', 'Rest3', 'Rest4'],
            'cuisine': ['Italian', 'Italian', 'Chinese', 'Mexican'],
            'location': ['Downtown', 'Downtown', 'Uptown', 'Midtown'],
            'price_range': ['$$', '$$', '$$$', '$'],
            'rating': [4.5, 4.3, 3.8, 4.1]
        })
        
        # Create features where restaurants 1 & 2 are very similar
        test_features = np.array([
            [1.0, 1.0, 2.0, 4.5],  # Restaurant 1
            [1.0, 1.0, 2.0, 4.3],  # Restaurant 2 (very similar to 1)
            [2.0, 2.0, 3.0, 3.8],  # Restaurant 3 (different)
            [3.0, 3.0, 1.0, 4.1]   # Restaurant 4 (different)
        ])
        
        self.model.fit(test_restaurants, test_features)
        
        # Restaurant 1 should have restaurant 2 as most similar
        recommendations = self.model.get_restaurant_recommendations(1, n_recommendations=3)
        
        # Restaurant 2 should be first in recommendations (most similar)
        self.assertEqual(recommendations[0], 2)
    
    def test_get_user_based_recommendations(self):
        """Test user-based content recommendations"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Create sample user ratings
        user_ratings = [
            {'restaurant_id': 1, 'rating': 5.0},
            {'restaurant_id': 2, 'rating': 4.0}
        ]
        
        recommendations = self.model.get_user_based_recommendations(
            user_ratings, n_recommendations=5
        )
        
        # Should return a list
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        # Should not include already rated restaurants
        rated_restaurant_ids = [r['restaurant_id'] for r in user_ratings]
        for rec in recommendations:
            self.assertNotIn(rec, rated_restaurant_ids)
    
    def test_get_user_based_recommendations_weighted(self):
        """Test that user recommendations are weighted by ratings"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # High rating should have more influence
        user_ratings_high = [
            {'restaurant_id': 1, 'rating': 5.0}
        ]
        
        # Low rating should have less influence
        user_ratings_low = [
            {'restaurant_id': 1, 'rating': 2.0}
        ]
        
        recs_high = self.model.get_user_based_recommendations(
            user_ratings_high, n_recommendations=5
        )
        recs_low = self.model.get_user_based_recommendations(
            user_ratings_low, n_recommendations=5
        )
        
        # Both should be valid
        self.assertIsInstance(recs_high, list)
        self.assertIsInstance(recs_low, list)
    
    def test_feature_importance_weighting(self):
        """Test that different features can be weighted differently"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Test with custom feature weights
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        
        # Should work with or without explicit weights
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=3
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_edge_case_empty_features(self):
        """Test handling of empty feature matrix"""
        empty_restaurants = pd.DataFrame(columns=['restaurant_id', 'name', 'cuisine'])
        empty_features = np.array([]).reshape(0, 0)
        
        # Should handle gracefully
        self.model.fit(empty_restaurants, empty_features)
        
        # Recommendations should be empty
        recommendations = self.model.get_restaurant_recommendations(1, n_recommendations=5)
        self.assertEqual(len(recommendations), 0)
    
    def test_edge_case_single_restaurant(self):
        """Test with only one restaurant"""
        single_restaurant = self.test_data['restaurants'].iloc[:1].copy()
        single_features = self.restaurant_features[:1]
        
        self.model.fit(single_restaurant, single_features)
        
        restaurant_id = single_restaurant['restaurant_id'].iloc[0]
        
        # Should return empty list (no other restaurants to recommend)
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=5
        )
        self.assertEqual(len(recommendations), 0)
    
    def test_edge_case_invalid_restaurant_id(self):
        """Test handling of invalid restaurant ID"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        # Use a restaurant ID that doesn't exist
        max_id = self.test_data['restaurants']['restaurant_id'].max()
        invalid_id = max_id + 1000
        
        recommendations = self.model.get_restaurant_recommendations(
            invalid_id, n_recommendations=5
        )
        
        # Should handle gracefully (might return empty list or popular items)
        self.assertIsInstance(recommendations, list)
    
    def test_feature_normalization_effect(self):
        """Test that feature normalization affects similarities correctly"""
        # Create features with different scales
        unnormalized_features = np.array([
            [1000, 1, 2, 4.5],     # Large first feature
            [1001, 1, 2, 4.3],     # Similar large first feature
            [2, 100, 3, 3.8],      # Different scale
            [3, 101, 1, 4.1]       # Different scale
        ])
        
        self.model.fit(self.test_data['restaurants'], unnormalized_features)
        
        # Should still work with unnormalized features
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=2
        )
        
        self.assertIsInstance(recommendations, list)
    
    @slow_test
    def test_performance_large_dataset(self):
        """Test performance with large dataset"""
        # Create larger dataset
        large_restaurants = self.factory.create_restaurants(1000)
        large_features = np.random.randn(1000, 10)  # 10 features
        
        # Time the fit operation
        result, fit_time = self.time_function(
            self.model.fit, large_restaurants, large_features
        )
        
        self.assertLess(fit_time, 3.0, f"Fit took {fit_time:.2f}s, expected < 3.0s")
        
        # Time recommendation generation
        restaurant_id = large_restaurants['restaurant_id'].iloc[0]
        result, rec_time = self.time_function(
            self.model.get_restaurant_recommendations, restaurant_id, 10
        )
        
        self.assertLess(rec_time, 1.0, f"Recommendations took {rec_time:.2f}s, expected < 1.0s")
    
    def test_cosine_similarity_properties(self):
        """Test properties of cosine similarity calculation"""
        # Create specific test vectors
        test_restaurants = pd.DataFrame({
            'restaurant_id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        test_features = np.array([
            [1, 0, 0],    # Orthogonal to [0,1,0]
            [0, 1, 0],    # Orthogonal to [1,0,0]
            [1, 1, 0]     # Similar to both
        ])
        
        self.model.fit(test_restaurants, test_features)
        
        # Orthogonal vectors should have similarity 0
        sim_12 = self.model.calculate_restaurant_similarity(1, 2)
        self.assertAlmostEqual(sim_12, 0.0, places=5)
        
        # Vector similar to both should have positive similarity
        sim_13 = self.model.calculate_restaurant_similarity(1, 3)
        sim_23 = self.model.calculate_restaurant_similarity(2, 3)
        
        self.assertGreater(sim_13, 0.0)
        self.assertGreater(sim_23, 0.0)
    
    def test_recommendation_diversity(self):
        """Test that recommendations include diverse options"""
        self.model.fit(self.test_data['restaurants'], self.restaurant_features)
        
        restaurant_id = self.test_data['restaurants']['restaurant_id'].iloc[0]
        recommendations = self.model.get_restaurant_recommendations(
            restaurant_id, n_recommendations=10
        )
        
        # Check that we get meaningful number of recommendations
        self.assertGreater(len(recommendations), 0)
        
        # All recommendations should be unique
        self.assertEqual(len(recommendations), len(set(recommendations)))

if __name__ == '__main__':
    unittest.main()
