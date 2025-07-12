#!/usr/bin/env python3
"""
Large-Scale Data Generator for AI Restaurant Recommendation System
Generates realistic restaurant, user, rating, and review data at scale
"""

import pandas as pd
import numpy as np
import random
import json
from faker import Faker
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fake = Faker()

class LargeScaleDataGenerator:
    """Generate large amounts of realistic restaurant data"""
    
    def __init__(self):
        self.cuisines = [
            'Italian', 'Chinese', 'Mexican', 'Indian', 'American', 'Japanese', 
            'Thai', 'French', 'Greek', 'Korean', 'Vietnamese', 'Mediterranean',
            'Spanish', 'Lebanese', 'Turkish', 'Brazilian', 'Ethiopian', 'Moroccan',
            'German', 'British', 'Russian', 'Persian', 'Filipino', 'Peruvian'
        ]
        
        self.price_ranges = ['$', '$$', '$$$', '$$$$']
        
        self.cities = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
            'Seattle', 'Denver', 'Washington', 'Boston', 'El Paso', 'Nashville',
            'Detroit', 'Oklahoma City', 'Portland', 'Las Vegas', 'Memphis', 'Louisville',
            'Baltimore', 'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento'
        ]
        
        self.restaurant_types = [
            'Fine Dining', 'Casual Dining', 'Fast Casual', 'Food Truck', 'Cafe',
            'Bistro', 'Buffet', 'Sports Bar', 'Wine Bar', 'Steakhouse', 'Seafood',
            'Vegetarian', 'Vegan', 'Organic', 'Farm-to-Table', 'Family Restaurant'
        ]
        
        self.positive_words = [
            'excellent', 'amazing', 'delicious', 'fantastic', 'wonderful', 'great',
            'outstanding', 'superb', 'incredible', 'perfect', 'brilliant', 'awesome',
            'fabulous', 'marvelous', 'spectacular', 'divine', 'exquisite', 'sublime'
        ]
        
        self.negative_words = [
            'terrible', 'awful', 'disgusting', 'horrible', 'worst', 'disappointing',
            'bland', 'overpriced', 'cold', 'stale', 'rude', 'slow', 'dirty', 'noisy'
        ]
        
        self.neutral_words = [
            'okay', 'average', 'decent', 'fine', 'acceptable', 'normal', 'standard',
            'typical', 'ordinary', 'reasonable', 'moderate', 'adequate', 'satisfactory'
        ]
    
    def generate_restaurants(self, n_restaurants=5000):
        """Generate large number of restaurants"""
        print(f"Generating {n_restaurants} restaurants...")
        
        restaurants = []
        for i in range(n_restaurants):
            restaurant_id = i + 1
            city = random.choice(self.cities)
            
            restaurant = {
                'restaurant_id': restaurant_id,
                'name': f"{fake.company()} {random.choice(['Restaurant', 'Bistro', 'Cafe', 'Kitchen', 'Grill', 'House'])}",
                'cuisine': random.choice(self.cuisines),
                'location': f"{fake.street_address()}, {city}, {fake.state_abbr()} {fake.zipcode()}",
                'city': city,
                'price_range': random.choice(self.price_ranges),
                'rating': round(random.uniform(2.5, 5.0), 1),
                'num_reviews': random.randint(10, 2000),
                'delivery_time': random.randint(15, 90),
                'is_open': np.random.choice([True, False], p=[0.85, 0.15]),
                'restaurant_type': random.choice(self.restaurant_types),
                'description': fake.text(max_nb_chars=300),
                'phone': fake.phone_number(),
                'website': fake.url(),
                'established_year': random.randint(1950, 2023),
                
                # Emotional attributes for advanced recommendations
                'comfort_level': round(random.uniform(0.0, 1.0), 3),
                'energy_level': round(random.uniform(0.0, 1.0), 3),
                'social_intimacy': round(random.uniform(0.0, 1.0), 3),
                'adventure_factor': round(random.uniform(0.0, 1.0), 3),
                'stress_relief': round(random.uniform(0.0, 1.0), 3),
                'romance_factor': round(random.uniform(0.0, 1.0), 3),
                
                # Business metrics
                'avg_wait_time': random.randint(5, 45),
                'capacity': random.randint(20, 200),
                'parking_available': np.random.choice([True, False], p=[0.7, 0.3]),
                'wheelchair_accessible': np.random.choice([True, False], p=[0.8, 0.2]),
                'accepts_reservations': np.random.choice([True, False], p=[0.6, 0.4]),
                'outdoor_seating': np.random.choice([True, False], p=[0.4, 0.6]),
                'wifi_available': np.random.choice([True, False], p=[0.9, 0.1]),
                'live_music': np.random.choice([True, False], p=[0.2, 0.8]),
                'happy_hour': np.random.choice([True, False], p=[0.4, 0.6])
            }
            
            restaurants.append(restaurant)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} restaurants...")
        
        return pd.DataFrame(restaurants)
    
    def generate_users(self, n_users=2000):
        """Generate large number of users"""
        print(f"Generating {n_users} users...")
        
        users = []
        for i in range(n_users):
            user = {
                'user_id': i + 1,
                'name': fake.name(),
                'email': fake.email(),
                'age': random.randint(18, 75),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'preferred_cuisine': random.choice(self.cuisines),
                'dietary_restrictions': random.choice(['None', 'Vegetarian', 'Vegan', 'Gluten-Free', 'Kosher', 'Halal']),
                'price_preference': random.choice(self.price_ranges),
                'location': random.choice(self.cities),
                'join_date': fake.date_between(start_date='-3y', end_date='today'),
                
                # User preferences and behavior
                'avg_rating': round(random.uniform(2.0, 5.0), 1),
                'total_reviews': random.randint(0, 100),
                'frequency': random.choice(['Daily', 'Weekly', 'Monthly', 'Occasional']),
                'group_size_preference': random.randint(1, 8),
                'time_preference': random.choice(['Breakfast', 'Lunch', 'Dinner', 'Late Night', 'Any']),
                'adventure_level': round(random.uniform(0.0, 1.0), 3),
                'social_level': round(random.uniform(0.0, 1.0), 3),
                'health_conscious': round(random.uniform(0.0, 1.0), 3),
                'budget_conscious': round(random.uniform(0.0, 1.0), 3)
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_ratings(self, users_df, restaurants_df, n_ratings=25000):
        """Generate realistic ratings with patterns"""
        print(f"Generating {n_ratings} ratings...")
        
        ratings = []
        for i in range(n_ratings):
            user = users_df.sample(1).iloc[0]
            restaurant = restaurants_df.sample(1).iloc[0]
            
            # Generate realistic rating based on user and restaurant characteristics
            base_rating = restaurant['rating']
            
            # Cuisine preference match
            cuisine_match = 1.0 if user['preferred_cuisine'] == restaurant['cuisine'] else 0.5
            
            # Price preference match
            user_price_idx = self.price_ranges.index(user['price_preference'])
            restaurant_price_idx = self.price_ranges.index(restaurant['price_range'])
            price_match = 1.0 - (abs(user_price_idx - restaurant_price_idx) * 0.2)
            
            # Location factor (same city bonus)
            location_match = 1.2 if user['location'] == restaurant['city'] else 1.0
            
            # Calculate final rating
            rating_adjustment = (cuisine_match * 0.4 + price_match * 0.3) * location_match
            final_rating = base_rating * rating_adjustment + random.uniform(-0.5, 0.5)
            final_rating = max(1.0, min(5.0, round(final_rating, 1)))
            
            ratings.append({
                'user_id': user['user_id'],
                'restaurant_id': restaurant['restaurant_id'],
                'rating': final_rating,
                'timestamp': fake.date_time_between(start_date='-2y', end_date='now'),
                'order_type': random.choice(['Dine-in', 'Takeout', 'Delivery']),
                'party_size': random.randint(1, 8),
                'wait_time': random.randint(5, 60),
                'occasion': random.choice(['Casual', 'Date', 'Business', 'Celebration', 'Family'])
            })
            
            if (i + 1) % 5000 == 0:
                print(f"Generated {i + 1} ratings...")
        
        return pd.DataFrame(ratings)
    
    def generate_reviews(self, ratings_df, n_reviews=15000):
        """Generate realistic reviews based on ratings"""
        print(f"Generating {n_reviews} reviews...")
        
        # Sample ratings for reviews
        sampled_ratings = ratings_df.sample(n=min(n_reviews, len(ratings_df)))
        
        reviews = []
        for i, (_, rating_row) in enumerate(sampled_ratings.iterrows()):
            rating = rating_row['rating']
            
            # Generate review based on rating
            if rating >= 4.0:
                sentiment = 'positive'
                word_list = self.positive_words
                base_text = "Great experience! The food was"
            elif rating <= 2.5:
                sentiment = 'negative'
                word_list = self.negative_words
                base_text = "Unfortunately, the food was"
            else:
                sentiment = 'neutral'
                word_list = self.neutral_words
                base_text = "The food was"
            
            # Create review text
            descriptive_words = random.sample(word_list, min(3, len(word_list)))
            review_text = f"{base_text} {' and '.join(descriptive_words)}. {fake.sentence()}"
            
            # Add more detail for longer reviews
            if random.random() > 0.5:
                review_text += f" {fake.sentence()}"
            
            if random.random() > 0.7:
                review_text += f" The service was {random.choice(word_list)}."
            
            reviews.append({
                'review_id': i + 1,
                'user_id': rating_row['user_id'],
                'restaurant_id': rating_row['restaurant_id'],
                'review_text': review_text,
                'rating': rating,
                'sentiment': sentiment,
                'timestamp': rating_row['timestamp'],
                'helpful_count': random.randint(0, 50),
                'verified_purchase': np.random.choice([True, False], p=[0.8, 0.2])
            })
            
            if (i + 1) % 3000 == 0:
                print(f"Generated {i + 1} reviews...")
        
        return pd.DataFrame(reviews)
    
    def save_all_data(self, data_dir='data', backup=True):
        """Generate and save all data"""
        print("🚀 Starting Large-Scale Data Generation")
        print("=" * 60)
        
        # Backup existing data
        if backup and os.path.exists(f'{data_dir}/restaurants.csv'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f'{data_dir}/backup_{timestamp}'
            os.makedirs(backup_dir, exist_ok=True)
            
            for file in ['restaurants.csv', 'users.csv', 'ratings.csv', 'reviews.csv']:
                if os.path.exists(f'{data_dir}/{file}'):
                    os.rename(f'{data_dir}/{file}', f'{backup_dir}/{file}')
            print(f"📦 Backed up existing data to {backup_dir}")
        
        # Generate data
        restaurants_df = self.generate_restaurants(5000)
        users_df = self.generate_users(2000)
        ratings_df = self.generate_ratings(users_df, restaurants_df, 25000)
        reviews_df = self.generate_reviews(ratings_df, 15000)
        
        # Save to CSV
        os.makedirs(data_dir, exist_ok=True)
        restaurants_df.to_csv(f'{data_dir}/restaurants.csv', index=False)
        users_df.to_csv(f'{data_dir}/users.csv', index=False)
        ratings_df.to_csv(f'{data_dir}/ratings.csv', index=False)
        reviews_df.to_csv(f'{data_dir}/reviews.csv', index=False)
        
        # Generate summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'restaurants': len(restaurants_df),
            'users': len(users_df),
            'ratings': len(ratings_df),
            'reviews': len(reviews_df),
            'cities': len(restaurants_df['city'].unique()),
            'cuisines': len(restaurants_df['cuisine'].unique()),
            'avg_rating': float(ratings_df['rating'].mean()),
            'rating_distribution': ratings_df['rating'].value_counts().to_dict()
        }
        
        with open(f'{data_dir}/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("=" * 60)
        print("✅ Data Generation Complete!")
        print(f"📊 Generated:")
        print(f"   • {len(restaurants_df):,} restaurants")
        print(f"   • {len(users_df):,} users")
        print(f"   • {len(ratings_df):,} ratings")
        print(f"   • {len(reviews_df):,} reviews")
        print(f"   • {len(restaurants_df['city'].unique())} cities")
        print(f"   • {len(restaurants_df['cuisine'].unique())} cuisines")
        print(f"📁 Saved to: {data_dir}/")
        
        return {
            'restaurants': restaurants_df,
            'users': users_df,
            'ratings': ratings_df,
            'reviews': reviews_df,
            'summary': summary
        }

def main():
    """Main data generation function"""
    generator = LargeScaleDataGenerator()
    data = generator.save_all_data()
    return data

if __name__ == "__main__":
    main()
