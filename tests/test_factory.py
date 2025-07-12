"""
Test Data Factory for AI Recommendation System

This module provides factory classes for generating realistic test data
for all components of the recommendation system including users, restaurants,
ratings, reviews, and emotional data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
from typing import Dict, List, Tuple, Optional
import json

fake = Faker()

class TestDataFactory:
    """Factory class for generating comprehensive test data"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducible tests"""
        np.random.seed(random_seed)
        random.seed(random_seed)
        Faker.seed(random_seed)
        
    def create_users(self, n_users: int = 100) -> pd.DataFrame:
        """Create realistic user data"""
        users_data = {
            'user_id': list(range(1, n_users + 1)),
            'age': np.random.randint(18, 70, n_users),
            'location': [fake.city() for _ in range(n_users)],
            'preferred_cuisine': np.random.choice([
                'Italian', 'Chinese', 'Mexican', 'Indian', 'American',
                'Japanese', 'Thai', 'French', 'Greek', 'Korean'
            ], n_users),
            'dietary_restrictions': [
                random.choice(['None', 'Vegetarian', 'Vegan', 'Gluten-Free', 'Keto'])
                for _ in range(n_users)
            ],
            'price_preference': np.random.choice(['$', '$$', '$$$', '$$$$'], n_users),
            'registration_date': [
                fake.date_between(start_date='-2y', end_date='today')
                for _ in range(n_users)
            ]
        }
        return pd.DataFrame(users_data)
    
    def create_restaurants(self, n_restaurants: int = 200) -> pd.DataFrame:
        """Create realistic restaurant data with emotional attributes"""
        restaurants_data = {
            'restaurant_id': list(range(1, n_restaurants + 1)),
            'name': [fake.company() + " Restaurant" for _ in range(n_restaurants)],
            'cuisine': np.random.choice([
                'Italian', 'Chinese', 'Mexican', 'Indian', 'American',
                'Japanese', 'Thai', 'French', 'Greek', 'Korean'
            ], n_restaurants),
            'location': [fake.address() for _ in range(n_restaurants)],
            'price_range': np.random.choice(['$', '$$', '$$$', '$$$$'], n_restaurants),
            'rating': np.random.uniform(3.0, 5.0, n_restaurants).round(1),
            'num_reviews': np.random.randint(10, 1000, n_restaurants),
            'delivery_time': np.random.randint(15, 60, n_restaurants),
            'is_open': np.random.choice([True, False], n_restaurants, p=[0.8, 0.2]),
            'description': [fake.text(max_nb_chars=200) for _ in range(n_restaurants)]
        }
        
        # Add emotional attributes
        for i in range(n_restaurants):
            restaurants_data.setdefault('comfort_level', []).append(np.random.uniform(0.0, 1.0))
            restaurants_data.setdefault('energy_level', []).append(np.random.uniform(0.0, 1.0))
            restaurants_data.setdefault('social_intimacy', []).append(np.random.uniform(0.0, 1.0))
            restaurants_data.setdefault('adventure_factor', []).append(np.random.uniform(0.0, 1.0))
            restaurants_data.setdefault('stress_relief', []).append(np.random.uniform(0.0, 1.0))
            restaurants_data.setdefault('romance_factor', []).append(np.random.uniform(0.0, 1.0))
        
        return pd.DataFrame(restaurants_data)
    
    def create_ratings(self, users_df: pd.DataFrame, restaurants_df: pd.DataFrame,
                      n_ratings: int = 2000) -> pd.DataFrame:
        """Create realistic rating data with patterns"""
        ratings_data = []
        
        for _ in range(n_ratings):
            user_id = random.choice(users_df['user_id'].tolist())
            restaurant_id = random.choice(restaurants_df['restaurant_id'].tolist())
            
            # Get user and restaurant info for realistic rating generation
            user = users_df[users_df['user_id'] == user_id].iloc[0]
            restaurant = restaurants_df[restaurants_df['restaurant_id'] == restaurant_id].iloc[0]
            
            # Generate rating based on cuisine preference match
            base_rating = restaurant['rating']
            if user['preferred_cuisine'] == restaurant['cuisine']:
                rating = min(5.0, base_rating + np.random.normal(0.5, 0.3))
            else:
                rating = max(1.0, base_rating + np.random.normal(-0.3, 0.5))
            
            ratings_data.append({
                'user_id': user_id,
                'restaurant_id': restaurant_id,
                'rating': round(max(1.0, min(5.0, rating)), 1),
                'timestamp': fake.date_time_between(start_date='-1y', end_date='now')
            })
        
        return pd.DataFrame(ratings_data)
    
    def create_reviews(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic review data with sentiment"""
        reviews_data = []
        
        positive_words = ['excellent', 'amazing', 'delicious', 'fantastic', 'wonderful', 'great']
        negative_words = ['terrible', 'awful', 'disgusting', 'horrible', 'worst', 'disappointing']
        neutral_words = ['okay', 'average', 'decent', 'fine', 'acceptable', 'normal']
        
        for _, rating_row in ratings_df.iterrows():
            rating = rating_row['rating']
            
            # Generate review based on rating
            if rating >= 4.0:
                review_words = random.choices(positive_words, k=random.randint(3, 8))
                sentiment = 'positive'
            elif rating <= 2.0:
                review_words = random.choices(negative_words, k=random.randint(3, 8))
                sentiment = 'negative'
            else:
                review_words = random.choices(neutral_words, k=random.randint(2, 6))
                sentiment = 'neutral'
            
            review_text = f"The food was {' and '.join(review_words)}. " + fake.sentence()
            
            reviews_data.append({
                'review_id': len(reviews_data) + 1,
                'user_id': rating_row['user_id'],
                'restaurant_id': rating_row['restaurant_id'],
                'review_text': review_text,
                'rating': rating_row['rating'],  # Add the rating column
                'sentiment': sentiment,
                'timestamp': rating_row['timestamp']
            })
        
        return pd.DataFrame(reviews_data)
    
    def create_emotional_mappings(self) -> Dict:
        """Create emotion-cuisine mappings for testing"""
        return {
            "emotion_cuisine_mapping": {
                "happy": ["Italian", "American", "Mexican"],
                "sad": ["Italian", "Indian", "comfort_food"],
                "stressed": ["Asian", "healthy", "quick_service"],
                "excited": ["Mexican", "Thai", "fusion"],
                "anxious": ["comfort_food", "familiar", "simple"],
                "romantic": ["Italian", "French", "fine_dining"],
                "energetic": ["Asian", "Mexican", "spicy"],
                "calm": ["Japanese", "healthy", "light"],
                "adventurous": ["fusion", "exotic", "ethnic"],
                "nostalgic": ["comfort_food", "traditional", "home_style"]
            },
            "restaurant_mood_attributes": {
                "comfort_level": "How comforting and cozy the restaurant feels",
                "energy_level": "How energetic and lively the atmosphere is",
                "social_intimacy": "How suitable for intimate or social dining",
                "adventure_factor": "How unique or adventurous the cuisine is",
                "stress_relief": "How relaxing and stress-relieving the environment is",
                "romance_factor": "How romantic and suitable for dates"
            }
        }
    
    def create_emotional_test_scenarios(self) -> List[Dict]:
        """Create test scenarios for emotional intelligence testing"""
        return [
            {
                "user_input": "I'm feeling really stressed from work today",
                "expected_emotions": ["stressed", "anxious"],
                "expected_cuisines": ["Asian", "comfort_food"],
                "context": {"time_of_day": "evening", "weather": "rainy"}
            },
            {
                "user_input": "I'm so excited about my promotion!",
                "expected_emotions": ["happy", "excited"],
                "expected_cuisines": ["Mexican", "Italian"],
                "context": {"time_of_day": "lunch", "weather": "sunny"}
            },
            {
                "user_input": "Anniversary dinner tonight, want something special",
                "expected_emotions": ["romantic", "happy"],
                "expected_cuisines": ["Italian", "French"],
                "context": {"time_of_day": "dinner", "special_occasion": "anniversary"}
            },
            {
                "user_input": "Feeling adventurous, want to try something new",
                "expected_emotions": ["adventurous", "excited"],
                "expected_cuisines": ["fusion", "exotic"],
                "context": {"time_of_day": "dinner", "mood": "exploratory"}
            },
            {
                "user_input": "Just want something simple and comforting",
                "expected_emotions": ["calm", "nostalgic"],
                "expected_cuisines": ["comfort_food", "American"],
                "context": {"time_of_day": "evening", "energy_level": "low"}
            }
        ]
    
    def create_performance_test_data(self, scale_factor: int = 1) -> Dict[str, pd.DataFrame]:
        """Create large-scale data for performance testing"""
        base_users = 1000 * scale_factor
        base_restaurants = 500 * scale_factor
        base_ratings = 10000 * scale_factor
        
        users_df = self.create_users(base_users)
        restaurants_df = self.create_restaurants(base_restaurants)
        ratings_df = self.create_ratings(users_df, restaurants_df, base_ratings)
        reviews_df = self.create_reviews(ratings_df.sample(n=min(len(ratings_df), 5000 * scale_factor)))
        
        return {
            'users': users_df,
            'restaurants': restaurants_df,
            'ratings': ratings_df,
            'reviews': reviews_df
        }
    
    def save_test_data(self, output_dir: str = 'test_data') -> None:
        """Save all test data to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create standard test datasets
        users_df = self.create_users()
        restaurants_df = self.create_restaurants()
        ratings_df = self.create_ratings(users_df, restaurants_df)
        reviews_df = self.create_reviews(ratings_df)
        
        # Save to CSV
        users_df.to_csv(f"{output_dir}/test_users.csv", index=False)
        restaurants_df.to_csv(f"{output_dir}/test_restaurants.csv", index=False)
        ratings_df.to_csv(f"{output_dir}/test_ratings.csv", index=False)
        reviews_df.to_csv(f"{output_dir}/test_reviews.csv", index=False)
        
        # Save emotional mappings
        emotional_mappings = self.create_emotional_mappings()
        with open(f"{output_dir}/test_emotion_mappings.json", 'w') as f:
            json.dump(emotional_mappings, f, indent=2)
        
        # Save test scenarios
        test_scenarios = self.create_emotional_test_scenarios()
        with open(f"{output_dir}/test_scenarios.json", 'w') as f:
            json.dump(test_scenarios, f, indent=2)
        
        print(f"Test data saved to {output_dir}/")

    def _create_feature_matrix(self, restaurants_df):
        """Create feature matrix for content-based filtering"""
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical features
        le_cuisine = LabelEncoder()
        le_price = LabelEncoder()
        
        cuisine_encoded = le_cuisine.fit_transform(restaurants_df['cuisine'])
        price_encoded = le_price.fit_transform(restaurants_df['price_range'])
        
        # Simplified location encoding (just use hash of first word)
        location_simplified = restaurants_df['location'].str.split().str[0].fillna('Unknown')
        location_encoded = pd.Categorical(location_simplified).codes
        
        # Create feature matrix with numerical features
        feature_matrix = np.column_stack([
            cuisine_encoded,
            location_encoded,
            price_encoded,
            restaurants_df['rating'].values
        ])
        
        return feature_matrix
    
# Utility functions for tests
def create_mock_user_item_matrix(n_users: int = 50, n_restaurants: int = 100, 
                                sparsity: float = 0.9) -> pd.DataFrame:
    """Create a sparse user-item matrix for testing"""
    matrix = np.zeros((n_users, n_restaurants))
    n_ratings = int(n_users * n_restaurants * (1 - sparsity))
    
    for _ in range(n_ratings):
        user_idx = np.random.randint(0, n_users)
        restaurant_idx = np.random.randint(0, n_restaurants)
        rating = np.random.randint(1, 6)
        matrix[user_idx, restaurant_idx] = rating
    
    return pd.DataFrame(matrix, 
                       index=range(1, n_users + 1),
                       columns=range(1, n_restaurants + 1))

def create_test_config() -> Dict:
    """Create test configuration overrides"""
    return {
        'USE_LLM_ENHANCEMENT': False,  # Disable for faster testing
        'USE_EMOTIONAL_RECOMMENDATIONS': True,
        'EMOTIONAL_CACHE_TTL': 60,  # Short cache for testing
        'N_RECOMMENDATIONS': 5,  # Fewer recommendations for testing
        'TEST_MODE': True
    }

if __name__ == "__main__":
    # Generate and save test data
    factory = TestDataFactory()
    factory.save_test_data()
    
    # Generate performance test data
    perf_data = factory.create_performance_test_data(scale_factor=2)
    print(f"Generated performance test data:")
    print(f"- Users: {len(perf_data['users'])}")
    print(f"- Restaurants: {len(perf_data['restaurants'])}")
    print(f"- Ratings: {len(perf_data['ratings'])}")
    print(f"- Reviews: {len(perf_data['reviews'])}")
