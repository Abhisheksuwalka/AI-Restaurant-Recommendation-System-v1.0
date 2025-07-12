import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_restaurant_data():
    """Generate sample restaurant data"""
    cuisines = ['Indian', 'Chinese', 'Italian', 'Mexican', 'Thai', 'Continental', 'South Indian', 'North Indian']
    locations = ['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout', 'BTM Layout', 'Jayanagar']
    
    restaurants = []
    for i in range(200):
        restaurant = {
            'restaurant_id': f'R{i+1:03d}',
            'name': f'Restaurant {i+1}',
            'cuisine': random.choice(cuisines),
            'location': random.choice(locations),
            'rating': round(random.uniform(3.0, 5.0), 1),
            'price_range': random.choice(['$', '$$', '$$$']),
            'delivery_time': random.randint(20, 60),
            'description': f'Authentic {random.choice(cuisines)} cuisine with great ambiance'
        }
        restaurants.append(restaurant)
    
    return pd.DataFrame(restaurants)

def generate_user_data():
    """Generate sample user data"""
    users = []
    for i in range(100):
        user = {
            'user_id': f'U{i+1:03d}',
            'age': random.randint(18, 60),
            'location': random.choice(['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout']),
            'preferred_cuisine': random.choice(['Indian', 'Chinese', 'Italian', 'Mexican']),
            'budget_preference': random.choice(['$', '$$', '$$$'])
        }
        users.append(user)
    
    return pd.DataFrame(users)

def generate_ratings_data(restaurants_df, users_df):
    """Generate user-restaurant ratings"""
    ratings = []
    for _ in range(2000):
        rating = {
            'user_id': random.choice(users_df['user_id'].tolist()),
            'restaurant_id': random.choice(restaurants_df['restaurant_id'].tolist()),
            'rating': random.randint(1, 5),
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 365))
        }
        ratings.append(rating)
    
    return pd.DataFrame(ratings)

def generate_reviews_data(ratings_df):
    """Generate sample reviews with sentiment"""
    positive_reviews = [
        "Amazing food quality and quick delivery!",
        "Loved the taste, will order again",
        "Excellent service and delicious food",
        "Great value for money",
        "Fresh ingredients and perfect packaging"
    ]
    
    negative_reviews = [
        "Food was cold when delivered",
        "Poor quality, not worth the price",
        "Late delivery and average taste",
        "Disappointing experience",
        "Food was too spicy and salty"
    ]
    
    neutral_reviews = [
        "Average food, nothing special",
        "Decent taste, could be better",
        "Okay experience, might try again",
        "Standard quality food",
        "Normal delivery time"
    ]
    
    reviews = []
    for _, row in ratings_df.iterrows():
        if row['rating'] >= 4:
            review_text = random.choice(positive_reviews)
        elif row['rating'] <= 2:
            review_text = random.choice(negative_reviews)
        else:
            review_text = random.choice(neutral_reviews)
        
        review = {
            'user_id': row['user_id'],
            'restaurant_id': row['restaurant_id'],
            'rating': row['rating'],
            'review_text': review_text,
            'timestamp': row['timestamp']
        }
        reviews.append(review)
    
    return pd.DataFrame(reviews)

# Generate and save data
if __name__ == "__main__":
    restaurants_df = generate_restaurant_data()
    users_df = generate_user_data()
    ratings_df = generate_ratings_data(restaurants_df, users_df)
    reviews_df = generate_reviews_data(ratings_df)
    
    # Save to CSV files
    restaurants_df.to_csv('data/restaurants.csv', index=False)
    users_df.to_csv('data/users.csv', index=False)
    ratings_df.to_csv('data/ratings.csv', index=False)
    reviews_df.to_csv('data/reviews.csv', index=False)
    
    print("Sample data generated successfully!")