"""
Performance Tests for AI Recommendation System

Tests system performance under various load conditions including
scalability, memory usage, response times, and stress testing.
"""

import unittest
import time
import psutil
import gc
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from tests.base_test import PerformanceTestCase, slow_test
from models.hybrid_recommender import HybridRecommender
from data.preprocessor import DataPreprocessor

class TestSystemPerformance(PerformanceTestCase):
    """Test system performance characteristics"""
    
    def setUp(self):
        super().setUp()
        self.large_data = self.factory.create_performance_test_data(scale_factor=2)
        
    def test_data_loading_performance(self):
        """Test data loading performance with large datasets"""
        preprocessor = DataPreprocessor()
        
        # Time data preprocessing
        start_time = time.time()
        
        # Simulate loading large data
        preprocessor.restaurants = self.large_data['restaurants']
        preprocessor.users = self.large_data['users'] 
        preprocessor.ratings = self.large_data['ratings']
        preprocessor.reviews = self.large_data['reviews']
        
        load_time = time.time() - start_time
        
        # Should load within reasonable time
        self.assertLess(load_time, 5.0, f"Data loading took {load_time:.2f}s")
        
        # Test preprocessing performance
        start_time = time.time()
        preprocessor.preprocess_restaurants()
        preprocess_time = time.time() - start_time
        
        self.assertLess(preprocess_time, 10.0, f"Preprocessing took {preprocess_time:.2f}s")
    
    def test_recommendation_generation_performance(self):
        """Test recommendation generation performance"""
        # Set up recommender with large data
        recommender = HybridRecommender()
        
        # Create processed data
        data = {
            'restaurants': self.large_data['restaurants'],
            'users': self.large_data['users'],
            'ratings': self.large_data['ratings'], 
            'reviews': self.large_data['reviews'],
            'user_item_matrix': self.large_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(self.large_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        # Time model training
        start_time = time.time()
        recommender.fit(data)
        fit_time = time.time() - start_time
        
        self.assertLess(fit_time, 30.0, f"Model fitting took {fit_time:.2f}s")
        
        # Test single recommendation performance
        user_id = data['users']['user_id'].iloc[0]
        
        start_time = time.time()
        recommendations = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
        rec_time = time.time() - start_time
        
        self.assertLess(rec_time, 2.0, f"Single recommendation took {rec_time:.2f}s")
        self.assertGreater(len(recommendations), 0)
    
    def test_concurrent_recommendation_performance(self):
        """Test performance under concurrent load"""
        # Set up smaller dataset for concurrent testing
        small_data = self.factory.create_performance_test_data(scale_factor=1)
        
        recommender = HybridRecommender()
        data = {
            'restaurants': small_data['restaurants'],
            'users': small_data['users'],
            'ratings': small_data['ratings'],
            'reviews': small_data['reviews'],
            'user_item_matrix': small_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(small_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        recommender.fit(data)
        
        # Test concurrent requests
        def get_recommendations(user_id):
            return recommender.get_hybrid_recommendations(user_id, n_recommendations=5)
        
        user_ids = small_data['users']['user_id'].iloc[:10].tolist()
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_recommendations, uid) for uid in user_ids]
            results = [future.result() for future in futures]
        
        concurrent_time = time.time() - start_time
        
        # Should handle concurrent requests efficiently
        self.assertLess(concurrent_time, 10.0, f"Concurrent requests took {concurrent_time:.2f}s")
        self.assertEqual(len(results), len(user_ids))
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load large dataset
        recommender = HybridRecommender()
        
        # Create large data
        large_data = self.factory.create_performance_test_data(scale_factor=3)
        
        data = {
            'restaurants': large_data['restaurants'],
            'users': large_data['users'],
            'ratings': large_data['ratings'],
            'reviews': large_data['reviews'],
            'user_item_matrix': large_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(large_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        # Fit model and check memory
        recommender.fit(data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Should not use excessive memory (threshold: 500MB increase)
        self.assertLess(memory_increase, 500, 
                       f"Memory usage increased by {memory_increase:.1f}MB")
        
        # Generate recommendations and check for memory leaks
        for i in range(10):
            user_id = data['users']['user_id'].iloc[i % len(data['users'])]
            recommender.get_hybrid_recommendations(user_id, n_recommendations=5)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_leak = final_memory - peak_memory
        
        # Should not have significant memory leaks (threshold: 50MB)
        self.assertLess(memory_leak, 50, 
                       f"Potential memory leak detected: {memory_leak:.1f}MB")
    
    def test_scalability_users(self):
        """Test scalability with increasing number of users"""
        scales = [100, 500, 1000]
        times = []
        
        for scale in scales:
            # Create data with different user counts
            test_data = self.factory.create_performance_test_data(scale_factor=scale/100)
            
            recommender = HybridRecommender()
            data = {
                'restaurants': test_data['restaurants'],
                'users': test_data['users'],
                'ratings': test_data['ratings'],
                'reviews': test_data['reviews'],
                'user_item_matrix': test_data['ratings'].pivot_table(
                    index='user_id', columns='restaurant_id', values='rating', fill_value=0
                ),
                'restaurant_features': np.random.randn(len(test_data['restaurants']), 4),
                'label_encoders': {}
            }
            
            # Time model fitting
            start_time = time.time()
            recommender.fit(data)
            fit_time = time.time() - start_time
            times.append(fit_time)
            
            # Should complete within reasonable time
            self.assertLess(fit_time, 60.0, 
                           f"Fit time for {scale} users: {fit_time:.2f}s")
        
        # Check that time complexity is reasonable (not exponential)
        # Time should not increase by more than 10x for 10x data increase
        if len(times) >= 2:
            time_ratio = times[-1] / times[0]
            data_ratio = scales[-1] / scales[0]
            
            self.assertLess(time_ratio, data_ratio * 2, 
                           "Time complexity appears to be worse than linear")
    
    def test_recommendation_response_time_distribution(self):
        """Test distribution of recommendation response times"""
        # Set up recommender
        test_data = self.factory.create_performance_test_data(scale_factor=1)
        
        recommender = HybridRecommender()
        data = {
            'restaurants': test_data['restaurants'],
            'users': test_data['users'],
            'ratings': test_data['ratings'],
            'reviews': test_data['reviews'],
            'user_item_matrix': test_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(test_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        recommender.fit(data)
        
        # Measure response times for multiple requests
        response_times = []
        user_ids = test_data['users']['user_id'].iloc[:50].tolist()
        
        for user_id in user_ids:
            start_time = time.time()
            recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        # Calculate statistics
        avg_time = np.mean(response_times)
        p95_time = np.percentile(response_times, 95)
        p99_time = np.percentile(response_times, 99)
        
        # Performance thresholds
        self.assertLess(avg_time, 1.0, f"Average response time: {avg_time:.3f}s")
        self.assertLess(p95_time, 2.0, f"95th percentile response time: {p95_time:.3f}s")
        self.assertLess(p99_time, 3.0, f"99th percentile response time: {p99_time:.3f}s")
    
    @slow_test
    def test_stress_testing(self):
        """Stress test the system with extreme loads"""
        # Create large dataset
        stress_data = self.factory.create_performance_test_data(scale_factor=5)
        
        recommender = HybridRecommender()
        data = {
            'restaurants': stress_data['restaurants'],
            'users': stress_data['users'],
            'ratings': stress_data['ratings'],
            'reviews': stress_data['reviews'][:10000],  # Limit reviews for stress test
            'user_item_matrix': stress_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(stress_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        # Should handle large data without crashing
        try:
            recommender.fit(data)
            
            # Generate many recommendations
            user_ids = stress_data['users']['user_id'].iloc[:100].tolist()
            
            for user_id in user_ids:
                recommendations = recommender.get_hybrid_recommendations(
                    user_id, n_recommendations=20
                )
                self.assertIsInstance(recommendations, list)
                
        except Exception as e:
            self.fail(f"Stress test failed: {str(e)}")
    
    def test_data_preprocessing_scalability(self):
        """Test data preprocessing scalability"""
        preprocessor = DataPreprocessor()
        
        # Test with different data sizes
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            # Create test data of specific size
            restaurants = self.factory.create_restaurants(size)
            
            start_time = time.time()
            
            preprocessor.restaurants = restaurants
            preprocessor.preprocess_restaurants()
            
            process_time = time.time() - start_time
            
            # Should scale reasonably (allow 0.01s per restaurant)
            max_time = size * 0.01
            self.assertLess(process_time, max_time,
                           f"Processing {size} restaurants took {process_time:.2f}s")
    
    def test_recommendation_quality_vs_speed_tradeoff(self):
        """Test tradeoff between recommendation quality and speed"""
        test_data = self.factory.create_performance_test_data(scale_factor=1)
        
        recommender = HybridRecommender()
        data = {
            'restaurants': test_data['restaurants'],
            'users': test_data['users'],
            'ratings': test_data['ratings'],
            'reviews': test_data['reviews'],
            'user_item_matrix': test_data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(test_data['restaurants']), 4),
            'label_encoders': {}
        }
        
        recommender.fit(data)
        
        user_id = data['users']['user_id'].iloc[0]
        
        # Test different recommendation counts
        for n_recs in [5, 10, 20, 50]:
            start_time = time.time()
            recommendations = recommender.get_hybrid_recommendations(
                user_id, n_recommendations=n_recs
            )
            rec_time = time.time() - start_time
            
            # Time should scale sub-linearly with number of recommendations
            expected_max_time = n_recs * 0.05  # 0.05s per recommendation
            self.assertLess(rec_time, expected_max_time,
                           f"Getting {n_recs} recommendations took {rec_time:.3f}s")
            
            # Should return requested number (or less if not available)
            self.assertLessEqual(len(recommendations), n_recs)

class TestMemoryProfiler(PerformanceTestCase):
    """Memory profiling tests"""
    
    def test_memory_efficiency(self):
        """Test memory efficiency of core operations"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create and process data
        data = self.factory.create_performance_test_data(scale_factor=1)
        
        recommender = HybridRecommender()
        processed_data = {
            'restaurants': data['restaurants'],
            'users': data['users'],
            'ratings': data['ratings'],
            'reviews': data['reviews'],
            'user_item_matrix': data['ratings'].pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(data['restaurants']), 4),
            'label_encoders': {}
        }
        
        recommender.fit(processed_data)
        
        # Generate recommendations
        user_id = data['users']['user_id'].iloc[0]
        for _ in range(10):
            recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        # Should use reasonable amount of memory
        self.assertLess(memory_used, 200, f"Used {memory_used:.1f}MB of memory")

if __name__ == '__main__':
    unittest.main()
