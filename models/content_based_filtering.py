import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class ContentBasedFiltering:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.restaurant_profiles = None
        self.content_similarity = None
        
    def fit(self, restaurants_df, restaurant_features):
        """Train the content-based filtering model"""
        self.restaurants = restaurants_df  # Store with expected name
        self.restaurants_df = restaurants_df  # Keep both for backward compatibility
        self.restaurant_features = restaurant_features
        
        # Create TF-IDF vectors for restaurant descriptions
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(restaurants_df['description'])
        
        # Combine numerical features with text features
        numerical_features = self.scaler.fit_transform(restaurant_features)
        
        # Combine features (you can adjust weights)
        text_weight = 0.3
        numerical_weight = 0.7
        
        self.restaurant_profiles = np.hstack([
            numerical_features * numerical_weight,
            tfidf_matrix.toarray() * text_weight
        ])
        
        # Calculate content similarity matrix
        self.content_similarity = cosine_similarity(self.restaurant_profiles)
        
        print("Content-based filtering model trained successfully!")
    
    def get_restaurant_recommendations(self, restaurant_id, n_recommendations=10):
        """Get similar restaurants based on content"""
        try:
            restaurant_idx = self.restaurants_df[
                self.restaurants_df['restaurant_id'] == restaurant_id
            ].index[0]
        except IndexError:
            return []
        
        # Get similarity scores
        similarity_scores = self.content_similarity[restaurant_idx]
        
        # Get indices of most similar restaurants
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            restaurant_info = self.restaurants_df.iloc[idx]
            recommendations.append({
                'restaurant_id': restaurant_info['restaurant_id'],
                'name': restaurant_info['name'],
                'cuisine': restaurant_info['cuisine'],
                'rating': restaurant_info['rating'],
                'similarity_score': similarity_scores[idx]
            })
        
        return recommendations
    
    def get_user_profile_recommendations(self, user_preferences, n_recommendations=10):
        """Get recommendations based on user preferences"""
        # Create user profile vector
        user_profile = self._create_user_profile(user_preferences)
        
        # Calculate similarity with all restaurants
        user_restaurant_similarity = cosine_similarity([user_profile], self.restaurant_profiles)[0]
        
        # Get top recommendations
        top_indices = user_restaurant_similarity.argsort()[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            restaurant_info = self.restaurants_df.iloc[idx]
            recommendations.append({
                'restaurant_id': restaurant_info['restaurant_id'],
                'name': restaurant_info['name'],
                'cuisine': restaurant_info['cuisine'],
                'rating': restaurant_info['rating'],
                'similarity_score': user_restaurant_similarity[idx]
            })
        
        return recommendations
    
    def _create_user_profile(self, user_preferences):
        """Create user profile vector from preferences"""
        # This is a simplified version - you can make it more sophisticated
        profile_vector = np.zeros(self.restaurant_profiles.shape[1])
        
        # Add logic to convert user preferences to feature vector
        # For now, using a simple approach
        
        return profile_vector
