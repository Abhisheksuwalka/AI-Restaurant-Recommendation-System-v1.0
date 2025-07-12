import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load all datasets"""
        self.restaurants = pd.read_csv('data/restaurants.csv')
        self.users = pd.read_csv('data/users.csv')
        self.ratings = pd.read_csv('data/ratings.csv')
        self.reviews = pd.read_csv('data/reviews.csv')
        
        print(f"Loaded {len(self.restaurants)} restaurants")
        print(f"Loaded {len(self.users)} users")
        print(f"Loaded {len(self.ratings)} ratings")
        print(f"Loaded {len(self.reviews)} reviews")
    
    def preprocess_restaurants(self):
        """Preprocess restaurant data"""
        # Encode categorical variables
        categorical_cols = ['cuisine', 'location', 'price_range']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.restaurants[f'{col}_encoded'] = le.fit_transform(self.restaurants[col])
            self.label_encoders[col] = le
        
        # Create feature matrix for content-based filtering
        feature_cols = ['cuisine_encoded', 'location_encoded', 'price_range_encoded', 'rating']
        self.restaurant_features = self.restaurants[feature_cols].values
        self.restaurant_features = self.scaler.fit_transform(self.restaurant_features)
        
        return self.restaurants
    
    def create_user_item_matrix(self):
        """Create user-item matrix for collaborative filtering"""
        user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='restaurant_id',
            values='rating',
            fill_value=0
        )
        
        return user_item_matrix
    
    def get_processed_data(self):
        """Return all processed data"""
        return {
            'restaurants': self.restaurants,
            'users': self.users,
            'ratings': self.ratings,
            'reviews': self.reviews,
            'user_item_matrix': self.create_user_item_matrix(),
            'restaurant_features': self.restaurant_features,
            'label_encoders': self.label_encoders
        }

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.preprocess_restaurants()
    data = preprocessor.get_processed_data()
    print("Data preprocessing completed!")
