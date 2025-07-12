"""
Fallback content-based filtering without sklearn for deployment
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class ContentBasedFilteringFallback:
    """Simplified content-based filtering without sklearn dependencies"""
    
    def __init__(self, restaurants_df: pd.DataFrame, reviews_df: pd.DataFrame):
        self.restaurants_df = restaurants_df.copy()
        self.reviews_df = reviews_df.copy()
        self.user_profiles = {}
        
    def prepare_data(self):
        """Prepare data for content-based filtering"""
        try:
            # Simple feature engineering without sklearn
            if 'cuisine_type' in self.restaurants_df.columns:
                # Create simple cuisine encoding
                cuisines = self.restaurants_df['cuisine_type'].unique()
                self.cuisine_mapping = {cuisine: i for i, cuisine in enumerate(cuisines)}
                
            # Aggregate ratings for restaurants
            if not self.reviews_df.empty:
                restaurant_stats = self.reviews_df.groupby('restaurant_id').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                restaurant_stats.columns = ['avg_rating', 'rating_count']
                
                self.restaurants_df = self.restaurants_df.merge(
                    restaurant_stats, 
                    left_on='restaurant_id', 
                    right_index=True, 
                    how='left'
                )
                
            # Fill missing values
            self.restaurants_df['avg_rating'] = self.restaurants_df.get('avg_rating', 4.0).fillna(4.0)
            self.restaurants_df['rating_count'] = self.restaurants_df.get('rating_count', 1).fillna(1)
            
            print("✅ Content-based filtering data prepared (fallback mode)")
            return True
            
        except Exception as e:
            print(f"⚠️ Error preparing content-based data: {e}")
            return False
    
    def build_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Build user preference profile"""
        try:
            user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
            
            if user_reviews.empty:
                return {'preferences': {}, 'avg_rating': 4.0}
            
            # Get restaurants user has rated
            user_restaurants = user_reviews.merge(
                self.restaurants_df, 
                on='restaurant_id', 
                how='left'
            )
            
            profile = {
                'avg_rating': user_reviews['rating'].mean(),
                'preferences': {}
            }
            
            # Cuisine preferences
            if 'cuisine_type' in user_restaurants.columns:
                cuisine_ratings = user_restaurants.groupby('cuisine_type')['rating'].mean()
                profile['preferences']['cuisine'] = cuisine_ratings.to_dict()
            
            return profile
            
        except Exception as e:
            print(f"⚠️ Error building user profile: {e}")
            return {'preferences': {}, 'avg_rating': 4.0}
    
    def recommend(self, user_id: str, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Generate content-based recommendations"""
        try:
            # Build user profile
            user_profile = self.build_user_profile(user_id)
            
            # Get restaurants user hasn't rated
            user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
            rated_restaurants = set(user_reviews['restaurant_id'].tolist())
            
            available_restaurants = self.restaurants_df[
                ~self.restaurants_df['restaurant_id'].isin(rated_restaurants)
            ].copy()
            
            if available_restaurants.empty:
                # Return top-rated restaurants as fallback
                top_restaurants = self.restaurants_df.nlargest(num_recommendations, 'avg_rating')
                return self._format_recommendations(top_restaurants, user_id)
            
            # Simple scoring based on user preferences
            scores = []
            
            for _, restaurant in available_restaurants.iterrows():
                score = restaurant.get('avg_rating', 4.0)
                
                # Boost score based on cuisine preference
                if 'cuisine' in user_profile['preferences']:
                    cuisine = restaurant.get('cuisine_type', 'Unknown')
                    if cuisine in user_profile['preferences']['cuisine']:
                        cuisine_score = user_profile['preferences']['cuisine'][cuisine]
                        score = score * 0.7 + cuisine_score * 0.3
                
                scores.append(score)
            
            available_restaurants['content_score'] = scores
            
            # Sort by score and return top recommendations
            recommendations = available_restaurants.nlargest(
                num_recommendations, 'content_score'
            )
            
            return self._format_recommendations(recommendations, user_id)
            
        except Exception as e:
            print(f"⚠️ Content-based recommendation error: {e}")
            # Return top-rated restaurants as fallback
            top_restaurants = self.restaurants_df.nlargest(num_recommendations, 'avg_rating')
            return self._format_recommendations(top_restaurants, user_id)
    
    def _format_recommendations(self, restaurants_df: pd.DataFrame, user_id: str) -> List[Dict[str, Any]]:
        """Format recommendations for output"""
        recommendations = []
        
        for _, restaurant in restaurants_df.iterrows():
            rec = {
                'restaurant_id': restaurant['restaurant_id'],
                'name': restaurant.get('name', 'Unknown Restaurant'),
                'cuisine_type': restaurant.get('cuisine_type', 'Unknown'),
                'avg_rating': float(restaurant.get('avg_rating', 4.0)),
                'rating_count': int(restaurant.get('rating_count', 1)),
                'recommendation_score': float(restaurant.get('content_score', restaurant.get('avg_rating', 4.0))),
                'method': 'content_based_fallback'
            }
            
            if 'address' in restaurant:
                rec['address'] = restaurant['address']
            if 'price_range' in restaurant:
                rec['price_range'] = restaurant['price_range']
                
            recommendations.append(rec)
        
        return recommendations
