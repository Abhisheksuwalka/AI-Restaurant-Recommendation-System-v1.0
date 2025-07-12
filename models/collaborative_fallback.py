"""
Fallback collaborative filtering without sklearn for deployment
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class CollaborativeFilteringFallback:
    """Simplified collaborative filtering without sklearn dependencies"""
    
    def __init__(self, ratings_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        self.ratings_df = ratings_df.copy()
        self.restaurants_df = restaurants_df.copy()
        self.user_item_matrix = None
        
    def prepare_data(self):
        """Prepare data for collaborative filtering"""
        try:
            # Create user-item matrix
            if not self.ratings_df.empty:
                self.user_item_matrix = self.ratings_df.pivot_table(
                    index='user_id',
                    columns='restaurant_id', 
                    values='rating',
                    fill_value=0
                )
            else:
                # Create empty matrix
                self.user_item_matrix = pd.DataFrame()
            
            print("✅ Collaborative filtering data prepared (fallback mode)")
            return True
            
        except Exception as e:
            print(f"⚠️ Error preparing collaborative data: {e}")
            return False
    
    def find_similar_users(self, user_id: str, num_similar: int = 10) -> List[str]:
        """Find similar users using simple correlation"""
        try:
            if user_id not in self.user_item_matrix.index:
                return []
            
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Simple similarity calculation
            similarities = []
            
            for other_user in self.user_item_matrix.index:
                if other_user == user_id:
                    continue
                
                other_ratings = self.user_item_matrix.loc[other_user]
                
                # Find common rated items
                common_items = (user_ratings > 0) & (other_ratings > 0)
                
                if common_items.sum() < 2:  # Need at least 2 common ratings
                    continue
                
                # Simple correlation calculation
                user_common = user_ratings[common_items]
                other_common = other_ratings[common_items]
                
                if user_common.std() == 0 or other_common.std() == 0:
                    similarity = 0
                else:
                    similarity = np.corrcoef(user_common, other_common)[0, 1]
                    if np.isnan(similarity):
                        similarity = 0
                
                similarities.append((other_user, similarity))
            
            # Sort by similarity and return top users
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [user for user, _ in similarities[:num_similar]]
            
        except Exception as e:
            print(f"⚠️ Error finding similar users: {e}")
            return []
    
    def recommend(self, user_id: str, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations"""
        try:
            if self.user_item_matrix.empty:
                return self._fallback_recommendations(num_recommendations)
            
            # Find similar users
            similar_users = self.find_similar_users(user_id)
            
            if not similar_users:
                return self._fallback_recommendations(num_recommendations)
            
            # Get items rated by user
            if user_id in self.user_item_matrix.index:
                user_ratings = self.user_item_matrix.loc[user_id]
                rated_items = set(user_ratings[user_ratings > 0].index)
            else:
                rated_items = set()
            
            # Aggregate recommendations from similar users
            item_scores = {}
            
            for similar_user in similar_users:
                similar_ratings = self.user_item_matrix.loc[similar_user]
                
                for item, rating in similar_ratings[similar_ratings > 0].items():
                    if item not in rated_items:  # Don't recommend already rated items
                        if item not in item_scores:
                            item_scores[item] = []
                        item_scores[item].append(rating)
            
            # Calculate average scores
            recommendations = []
            for item, ratings in item_scores.items():
                avg_score = np.mean(ratings)
                count = len(ratings)
                
                # Get restaurant details
                restaurant_info = self.restaurants_df[
                    self.restaurants_df['restaurant_id'] == item
                ]
                
                if not restaurant_info.empty:
                    restaurant = restaurant_info.iloc[0]
                    rec = {
                        'restaurant_id': item,
                        'name': restaurant.get('name', 'Unknown Restaurant'),
                        'cuisine_type': restaurant.get('cuisine_type', 'Unknown'),
                        'recommendation_score': float(avg_score),
                        'similar_users_count': count,
                        'method': 'collaborative_fallback'
                    }
                    
                    if 'address' in restaurant:
                        rec['address'] = restaurant['address']
                    if 'price_range' in restaurant:
                        rec['price_range'] = restaurant['price_range']
                    
                    recommendations.append(rec)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            print(f"⚠️ Collaborative filtering error: {e}")
            return self._fallback_recommendations(num_recommendations)
    
    def _fallback_recommendations(self, num_recommendations: int) -> List[Dict[str, Any]]:
        """Fallback to popular items when collaborative filtering fails"""
        try:
            # Calculate restaurant popularity
            if not self.ratings_df.empty:
                popularity = self.ratings_df.groupby('restaurant_id').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                popularity.columns = ['avg_rating', 'rating_count']
                
                # Combine with restaurant info
                popular_restaurants = self.restaurants_df.merge(
                    popularity, 
                    left_on='restaurant_id', 
                    right_index=True, 
                    how='left'
                )
                
                popular_restaurants['avg_rating'] = popular_restaurants['avg_rating'].fillna(4.0)
                popular_restaurants['rating_count'] = popular_restaurants['rating_count'].fillna(1)
                
                # Sort by popularity (rating * log(count))
                popular_restaurants['popularity_score'] = (
                    popular_restaurants['avg_rating'] * 
                    np.log1p(popular_restaurants['rating_count'])
                )
                
                top_restaurants = popular_restaurants.nlargest(
                    num_recommendations, 'popularity_score'
                )
            else:
                # If no ratings data, just return first N restaurants
                top_restaurants = self.restaurants_df.head(num_recommendations)
                top_restaurants['avg_rating'] = 4.0
                top_restaurants['rating_count'] = 1
                top_restaurants['popularity_score'] = 4.0
            
            recommendations = []
            for _, restaurant in top_restaurants.iterrows():
                rec = {
                    'restaurant_id': restaurant['restaurant_id'],
                    'name': restaurant.get('name', 'Unknown Restaurant'),
                    'cuisine_type': restaurant.get('cuisine_type', 'Unknown'),
                    'avg_rating': float(restaurant.get('avg_rating', 4.0)),
                    'rating_count': int(restaurant.get('rating_count', 1)),
                    'recommendation_score': float(restaurant.get('popularity_score', 4.0)),
                    'method': 'popularity_fallback'
                }
                
                if 'address' in restaurant:
                    rec['address'] = restaurant['address']
                if 'price_range' in restaurant:
                    rec['price_range'] = restaurant['price_range']
                    
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Fallback recommendations error: {e}")
            return []
