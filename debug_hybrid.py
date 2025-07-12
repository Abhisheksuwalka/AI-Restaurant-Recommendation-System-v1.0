#!/usr/bin/env python3
"""
Debug hybrid recommender issue
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_hybrid_recommender():
    """Debug the specific hybrid recommender issue"""
    try:
        from models.hybrid_recommender import HybridRecommender
        from tests.test_factory import TestDataFactory
        import numpy as np
        
        factory = TestDataFactory()
        recommender = HybridRecommender()
        
        # Create test data
        users = factory.create_users(30)
        restaurants = factory.create_restaurants(50)
        ratings = factory.create_ratings(users, restaurants, 200)
        reviews = factory.create_reviews(ratings.head(100))
        
        print("Test data created successfully:")
        print(f"Users: {len(users)} columns: {users.columns.tolist()}")
        print(f"Restaurants: {len(restaurants)} columns: {restaurants.columns.tolist()}")
        print(f"Ratings: {len(ratings)} columns: {ratings.columns.tolist()}")
        print(f"Reviews: {len(reviews)} columns: {reviews.columns.tolist()}")
        
        # Create processed data structure
        data = {
            'restaurants': restaurants,
            'users': users,
            'ratings': ratings,
            'reviews': reviews,
            'user_item_matrix': ratings.pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            ),
            'restaurant_features': np.random.randn(len(restaurants), 4),
            'label_encoders': {}
        }
        
        print("\nData structure created successfully")
        print(f"User-item matrix shape: {data['user_item_matrix'].shape}")
        print(f"Restaurant features shape: {data['restaurant_features'].shape}")
        
        # Test model fitting step by step
        print("\nTesting collaborative filtering...")
        recommender.collaborative_model.fit(data['user_item_matrix'])
        print("✓ Collaborative filtering OK")
        
        print("\nTesting content-based filtering...")
        recommender.content_model.fit(data['restaurants'], data['restaurant_features'])
        print("✓ Content-based filtering OK")
        
        print("\nTesting sentiment analysis...")
        print(f"Reviews sample:\n{reviews.head()}")
        
        # Check if reviews have all required columns
        required_cols = ['user_id', 'restaurant_id', 'review_text', 'rating', 'timestamp']
        missing_cols = [col for col in required_cols if col not in reviews.columns]
        if missing_cols:
            print(f"Missing columns in reviews: {missing_cols}")
        else:
            print("All required columns present in reviews")
            
        sentiment_results = recommender.sentiment_analyzer.analyze_reviews_batch(data['reviews'])
        print("✓ Sentiment analysis OK")
        print(f"Sentiment results columns: {sentiment_results.columns.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_hybrid_recommender()
