import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class CollaborativeFiltering:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_similarity = None
        self.item_similarity = None
        
    def fit(self, user_item_matrix):
        """Train the collaborative filtering model"""
        self.user_item_matrix = user_item_matrix
        self.users = user_item_matrix.index.tolist()
        self.items = user_item_matrix.columns.tolist()
        
        # Adaptive n_components based on data size
        max_components = min(user_item_matrix.shape) - 1
        adaptive_components = min(self.n_components, max_components)
        
        # Calculate user-user similarity
        user_similarity_matrix = cosine_similarity(user_item_matrix.values)
        self.user_similarity = pd.DataFrame(
            user_similarity_matrix,
            index=self.users,
            columns=self.users
        )
        
        # Calculate item-item similarity
        item_similarity_matrix = cosine_similarity(user_item_matrix.T.values)
        self.item_similarity = pd.DataFrame(
            item_similarity_matrix,
            index=self.items,
            columns=self.items
        )
        
        # Apply SVD for dimensionality reduction with adaptive components
        if adaptive_components > 0:
            self.svd = TruncatedSVD(n_components=adaptive_components, random_state=42)
            self.user_factors = self.svd.fit_transform(user_item_matrix.values)
            self.item_factors = self.svd.components_.T
        else:
            # Fallback for very small datasets
            self.user_factors = user_item_matrix.values
            self.item_factors = user_item_matrix.T.values
        
        print(f"Collaborative filtering model trained successfully with {adaptive_components} components!")
    
    def predict_rating(self, user_id, restaurant_id):
        """Predict rating for a user-restaurant pair"""
        if user_id not in self.users or restaurant_id not in self.items:
            return 3.0  # Default rating
        
        user_idx = self.users.index(user_id)
        item_idx = self.items.index(restaurant_id)
        
        # SVD-based prediction
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Normalize to 1-5 scale
        predicted_rating = max(1, min(5, predicted_rating + 3))
        
        return predicted_rating
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations for a specific user"""
        if user_id not in self.users:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings for unrated items
        predictions = []
        for item in unrated_items:
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append((item, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id, n_users=5):
        """Find similar users"""
        if user_id not in self.users:
            return []
        
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:n_users+1]
        return similar_users.index.tolist()
    
    def get_similar_items(self, restaurant_id, n_items=5):
        """Find similar restaurants"""
        if restaurant_id not in self.items:
            return []
        
        similar_items = self.item_similarity[restaurant_id].sort_values(ascending=False)[1:n_items+1]
        return similar_items.index.tolist()
