"""
Comprehensive Tests for Collaborative Filtering

Tests the collaborative filtering recommendation algorithm including
similarity calculations, recommendation generation, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from tests.base_test import ModelTestCase, slow_test
from models.collaborative_filtering import CollaborativeFiltering

class TestCollaborativeFiltering(ModelTestCase):
    """Test the CollaborativeFiltering model"""
    
    def setUp(self):
        super().setUp()
        self.model = CollaborativeFiltering()
    
    def test_fit_method_exists(self):
        """Test that fit method exists and accepts user-item matrix"""
        self.assertTrue(hasattr(self.model, 'fit'))
        
        # Should not raise exception
        self.model.fit(self.user_item_matrix)
        
        # Check if user_item_matrix is stored
        self.assertIsNotNone(self.model.user_item_matrix)
    
    def test_fit_stores_matrix_correctly(self):
        """Test that fit method stores the user-item matrix correctly"""
        self.model.fit(self.user_item_matrix)
        
        # Check if the stored matrix matches the input
        pd.testing.assert_frame_equal(self.model.user_item_matrix, self.user_item_matrix)
    
    def test_user_similarity_calculation(self):
        """Test user similarity calculation"""
        self.model.fit(self.user_item_matrix)
        
        # Get two users from the matrix
        user_ids = list(self.user_item_matrix.index)
        if len(user_ids) >= 2:
            user1, user2 = user_ids[0], user_ids[1]
            
            # Calculate similarity
            similarity = self.model.calculate_user_similarity(user1, user2)
            
            # Similarity should be between -1 and 1
            self.assertGreaterEqual(similarity, -1.0)
            self.assertLessEqual(similarity, 1.0)
            self.assertIsInstance(similarity, (float, np.floating))
    
    def test_user_similarity_identical_users(self):
        """Test that identical users have similarity of 1.0"""
        # Create a user-item matrix with identical preferences
        identical_matrix = pd.DataFrame({
            1: [5, 4, 3, 0, 0],
            2: [4, 5, 2, 0, 0],
            3: [3, 4, 5, 0, 0]
        }, index=[1, 2])
        
        self.model.fit(identical_matrix)
        
        # Same user should have similarity 1.0 with itself
        similarity = self.model.calculate_user_similarity(1, 1)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_user_similarity_no_common_ratings(self):
        """Test similarity when users have no common ratings"""
        # Create matrix where users have no overlapping ratings
        no_overlap_matrix = pd.DataFrame({
            1: [5, 4, 0, 0],
            2: [0, 0, 4, 5],
            3: [3, 0, 0, 2],
            4: [0, 3, 5, 0]
        }, index=[1, 2])
        
        self.model.fit(no_overlap_matrix)
        
        # Users with no common ratings should have similarity 0
        similarity = self.model.calculate_user_similarity(1, 2)
        self.assertEqual(similarity, 0.0)
    
    def test_get_user_recommendations_basic(self):
        """Test basic recommendation generation"""
        self.model.fit(self.user_item_matrix)
        
        # Get a user that exists in the matrix
        user_ids = list(self.user_item_matrix.index)
        if user_ids:
            user_id = user_ids[0]
            recommendations = self.model.get_user_recommendations(user_id, n_recommendations=5)
            
            # Should return a list
            self.assertIsInstance(recommendations, list)
            
            # Should not exceed requested number
            self.assertLessEqual(len(recommendations), 5)
            
            # Each recommendation should be a restaurant ID
            for rec in recommendations:
                self.assertIsInstance(rec, (int, np.integer))
                self.assertIn(rec, self.user_item_matrix.columns)
    
    def test_get_user_recommendations_excludes_rated(self):
        """Test that recommendations exclude already rated items"""
        # Create specific test case
        test_matrix = pd.DataFrame({
            1: [5, 0, 0],
            2: [4, 5, 0], 
            3: [0, 4, 5]
        }, index=[1, 2])
        
        self.model.fit(test_matrix)
        
        # User 1 has rated restaurant 1, so it shouldn't be recommended
        recommendations = self.model.get_user_recommendations(1, n_recommendations=5)
        
        # Should not include restaurant 1 (already rated)
        self.assertNotIn(1, recommendations)
    
    def test_get_user_recommendations_new_user(self):
        """Test recommendations for a user not in training data"""
        self.model.fit(self.user_item_matrix)
        
        # Use a user ID that doesn't exist
        max_user_id = max(self.user_item_matrix.index)
        new_user_id = max_user_id + 1
        
        recommendations = self.model.get_user_recommendations(new_user_id, n_recommendations=5)
        
        # Should return popular items or handle gracefully
        # Implementation may vary, but should not crash
        self.assertIsInstance(recommendations, list)
    
    def test_predict_rating_method(self):
        """Test rating prediction for user-item pairs"""
        self.model.fit(self.user_item_matrix)
        
        user_ids = list(self.user_item_matrix.index)
        restaurant_ids = list(self.user_item_matrix.columns)
        
        if user_ids and restaurant_ids:
            user_id = user_ids[0]
            restaurant_id = restaurant_ids[0]
            
            predicted_rating = self.model.predict_rating(user_id, restaurant_id)
            
            # Should return a valid rating
            self.assertIsInstance(predicted_rating, (float, np.floating))
            self.assertGreaterEqual(predicted_rating, 0.0)
            self.assertLessEqual(predicted_rating, 5.0)
    
    def test_recommendations_consistency(self):
        """Test that recommendations are consistent across multiple calls"""
        self.model.fit(self.user_item_matrix)
        
        user_ids = list(self.user_item_matrix.index)
        if user_ids:
            user_id = user_ids[0]
            
            # Get recommendations twice
            recs1 = self.model.get_user_recommendations(user_id, n_recommendations=5)
            recs2 = self.model.get_user_recommendations(user_id, n_recommendations=5)
            
            # Should be the same (deterministic)
            self.assertEqual(recs1, recs2)
    
    def test_edge_case_empty_matrix(self):
        """Test handling of empty user-item matrix"""
        empty_matrix = pd.DataFrame()
        
        # Should handle gracefully
        self.model.fit(empty_matrix)
        
        # Recommendations for any user should be empty
        recommendations = self.model.get_user_recommendations(1, n_recommendations=5)
        self.assertEqual(len(recommendations), 0)
    
    def test_edge_case_single_user(self):
        """Test with matrix containing only one user"""
        single_user_matrix = pd.DataFrame({
            1: [5], 2: [4], 3: [3]
        }, index=[1])
        
        self.model.fit(single_user_matrix)
        
        # Should handle gracefully
        recommendations = self.model.get_user_recommendations(1, n_recommendations=2)
        self.assertIsInstance(recommendations, list)
    
    def test_edge_case_all_zeros(self):
        """Test with matrix containing all zeros"""
        zero_matrix = pd.DataFrame(np.zeros((5, 10)), 
                                  index=range(1, 6), 
                                  columns=range(1, 11))
        
        self.model.fit(zero_matrix)
        
        # Should handle gracefully
        recommendations = self.model.get_user_recommendations(1, n_recommendations=5)
        self.assertIsInstance(recommendations, list)
    
    @slow_test
    def test_performance_large_matrix(self):
        """Test performance with large user-item matrix"""
        # Create larger matrix for performance testing
        large_matrix = pd.DataFrame(
            np.random.randint(0, 6, size=(1000, 500)),
            index=range(1, 1001),
            columns=range(1, 501)
        )
        
        # Time the fit operation
        result, fit_time = self.time_function(self.model.fit, large_matrix)
        
        # Should complete within reasonable time
        self.assertLess(fit_time, 5.0, f"Fit took {fit_time:.2f}s, expected < 5.0s")
        
        # Time recommendation generation
        result, rec_time = self.time_function(
            self.model.get_user_recommendations, 1, 10
        )
        
        self.assertLess(rec_time, 2.0, f"Recommendations took {rec_time:.2f}s, expected < 2.0s")
    
    def test_similarity_calculation_edge_cases(self):
        """Test similarity calculation with edge cases"""
        # Test with very sparse data
        sparse_matrix = pd.DataFrame({
            1: [5, 0, 0, 0, 0],
            2: [0, 4, 0, 0, 0],
            3: [0, 0, 3, 0, 0],
            4: [0, 0, 0, 2, 0],
            5: [0, 0, 0, 0, 1]
        }, index=[1, 2, 3])
        
        self.model.fit(sparse_matrix)
        
        # Calculate similarities
        sim_12 = self.model.calculate_user_similarity(1, 2)
        sim_13 = self.model.calculate_user_similarity(1, 3)
        
        # Should handle sparse data gracefully
        self.assertIsInstance(sim_12, (float, np.floating))
        self.assertIsInstance(sim_13, (float, np.floating))
    
    def test_recommendation_ranking(self):
        """Test that recommendations are properly ranked"""
        # Create matrix where we can predict the ranking
        test_matrix = pd.DataFrame({
            1: [5, 4, 0, 0],  # User 1 likes restaurant 1 & 2
            2: [4, 5, 0, 0],  # User 2 likes restaurant 1 & 2  
            3: [0, 0, 5, 4],  # User 3 likes restaurant 3 & 4
            4: [0, 0, 4, 5]   # User 4 likes restaurant 3 & 4
        }, index=[1, 2, 3])
        
        self.model.fit(test_matrix)
        
        # User 1 should get recommendations for restaurants 3 & 4
        # based on similarity with user 3
        recommendations = self.model.get_user_recommendations(1, n_recommendations=2)
        
        # Should contain unrated restaurants
        for rec in recommendations:
            self.assertIn(rec, [3, 4])  # Only unrated restaurants

if __name__ == '__main__':
    unittest.main()
