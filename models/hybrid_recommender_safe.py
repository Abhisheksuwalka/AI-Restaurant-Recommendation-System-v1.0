"""
Deployment-safe hybrid recommender that falls back when sklearn is not available
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os

class HybridRecommenderSafe:
    """Hybrid recommender with fallback support for deployment"""
    
    def __init__(self, 
                 restaurants_df: pd.DataFrame,
                 users_df: pd.DataFrame, 
                 ratings_df: pd.DataFrame,
                 reviews_df: Optional[pd.DataFrame] = None):
        
        self.restaurants_df = restaurants_df.copy()
        self.users_df = users_df.copy() 
        self.ratings_df = ratings_df.copy()
        self.reviews_df = reviews_df.copy() if reviews_df is not None else pd.DataFrame()
        
        self.collaborative_model = None
        self.content_model = None
        self.sentiment_analyzer = None
        self.is_sklearn_available = self._check_sklearn()
        
        self._initialize_models()
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available"""
        try:
            import sklearn
            return True
        except ImportError:
            print("⚠️ scikit-learn not available, using fallback models")
            return False
    
    def _initialize_models(self):
        """Initialize recommendation models"""
        try:
            if self.is_sklearn_available:
                # Use full models if sklearn is available
                from .collaborative_filtering import CollaborativeFiltering
                from .content_based_filtering import ContentBasedFiltering
                
                self.collaborative_model = CollaborativeFiltering(
                    self.ratings_df, self.restaurants_df
                )
                self.content_model = ContentBasedFiltering(
                    self.restaurants_df, self.reviews_df
                )
                print("✅ Using full sklearn-based models")
                
            else:
                # Use fallback models
                from .collaborative_fallback import CollaborativeFilteringFallback
                from .content_based_fallback import ContentBasedFilteringFallback
                
                self.collaborative_model = CollaborativeFilteringFallback(
                    self.ratings_df, self.restaurants_df
                )
                self.content_model = ContentBasedFilteringFallback(
                    self.restaurants_df, self.reviews_df
                )
                print("✅ Using fallback models (no sklearn)")
            
            # Initialize sentiment analyzer if available
            try:
                from .sentiment_analyzer import SentimentAnalyzer
                self.sentiment_analyzer = SentimentAnalyzer()
                print("✅ Sentiment analyzer initialized")
            except Exception as e:
                print(f"⚠️ Sentiment analyzer not available: {e}")
                self.sentiment_analyzer = None
                
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            self.collaborative_model = None
            self.content_model = None
    
    def prepare_data(self) -> bool:
        """Prepare data for all models"""
        try:
            success = True
            
            if self.collaborative_model:
                success &= self.collaborative_model.prepare_data()
            
            if self.content_model:
                success &= self.content_model.prepare_data()
            
            if self.sentiment_analyzer:
                try:
                    self.sentiment_analyzer.prepare_data()
                except Exception as e:
                    print(f"⚠️ Sentiment analyzer preparation failed: {e}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error preparing hybrid model data: {e}")
            return False
    
    def get_recommendations(self, 
                          user_id: str, 
                          num_recommendations: int = 10,
                          collaborative_weight: float = 0.5,
                          content_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Get hybrid recommendations"""
        try:
            recommendations = []
            
            # Get collaborative recommendations
            collab_recs = []
            if self.collaborative_model:
                try:
                    collab_recs = self.collaborative_model.recommend(
                        user_id, num_recommendations * 2
                    )
                except Exception as e:
                    print(f"⚠️ Collaborative filtering failed: {e}")
            
            # Get content-based recommendations  
            content_recs = []
            if self.content_model:
                try:
                    content_recs = self.content_model.recommend(
                        user_id, num_recommendations * 2
                    )
                except Exception as e:
                    print(f"⚠️ Content-based filtering failed: {e}")
            
            # Combine recommendations
            if collab_recs and content_recs:
                recommendations = self._hybrid_combine(
                    collab_recs, content_recs, 
                    collaborative_weight, content_weight
                )
            elif collab_recs:
                recommendations = collab_recs
            elif content_recs:
                recommendations = content_recs
            else:
                # Final fallback to popular restaurants
                recommendations = self._get_popular_restaurants(num_recommendations)
            
            # Enhance with sentiment if available
            if self.sentiment_analyzer and self.reviews_df is not None:
                try:
                    recommendations = self._enhance_with_sentiment(recommendations)
                except Exception as e:
                    print(f"⚠️ Sentiment enhancement failed: {e}")
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            print(f"❌ Error generating hybrid recommendations: {e}")
            return self._get_popular_restaurants(num_recommendations)
    
    def _hybrid_combine(self, 
                       collab_recs: List[Dict], 
                       content_recs: List[Dict],
                       collab_weight: float, 
                       content_weight: float) -> List[Dict[str, Any]]:
        """Combine collaborative and content-based recommendations"""
        try:
            # Create lookup for content recommendations
            content_lookup = {
                rec['restaurant_id']: rec['recommendation_score'] 
                for rec in content_recs
            }
            
            # Create lookup for collaborative recommendations
            collab_lookup = {
                rec['restaurant_id']: rec['recommendation_score'] 
                for rec in collab_recs
            }
            
            # Get all unique restaurants
            all_restaurants = set()
            all_restaurants.update(content_lookup.keys())
            all_restaurants.update(collab_lookup.keys())
            
            # Calculate hybrid scores
            hybrid_recs = []
            
            for restaurant_id in all_restaurants:
                collab_score = collab_lookup.get(restaurant_id, 0)
                content_score = content_lookup.get(restaurant_id, 0)
                
                # Normalize scores (simple min-max)
                if collab_score > 0:
                    collab_score = min(collab_score / 5.0, 1.0)  # Assuming 5-star scale
                if content_score > 0:
                    content_score = min(content_score / 5.0, 1.0)
                
                hybrid_score = (
                    collab_weight * collab_score + 
                    content_weight * content_score
                )
                
                # Get restaurant details (prefer content rec details)
                if restaurant_id in [r['restaurant_id'] for r in content_recs]:
                    base_rec = next(r for r in content_recs if r['restaurant_id'] == restaurant_id)
                else:
                    base_rec = next(r for r in collab_recs if r['restaurant_id'] == restaurant_id)
                
                hybrid_rec = base_rec.copy()
                hybrid_rec.update({
                    'recommendation_score': hybrid_score,
                    'collaborative_score': collab_score,
                    'content_score': content_score,
                    'method': 'hybrid'
                })
                
                hybrid_recs.append(hybrid_rec)
            
            # Sort by hybrid score
            hybrid_recs.sort(key=lambda x: x['recommendation_score'], reverse=True)
            return hybrid_recs
            
        except Exception as e:
            print(f"⚠️ Error combining recommendations: {e}")
            return collab_recs + content_recs
    
    def _enhance_with_sentiment(self, recommendations: List[Dict]) -> List[Dict[str, Any]]:
        """Enhance recommendations with sentiment analysis"""
        try:
            for rec in recommendations:
                restaurant_id = rec['restaurant_id']
                
                # Get reviews for this restaurant
                restaurant_reviews = self.reviews_df[
                    self.reviews_df['restaurant_id'] == restaurant_id
                ]
                
                if not restaurant_reviews.empty:
                    # Analyze sentiment
                    sentiments = []
                    for review_text in restaurant_reviews['review_text'].dropna():
                        sentiment = self.sentiment_analyzer.analyze_sentiment(review_text)
                        sentiments.append(sentiment.get('compound', 0))
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        rec['sentiment_score'] = avg_sentiment
                        
                        # Adjust recommendation score based on sentiment
                        sentiment_boost = max(0, avg_sentiment) * 0.1  # Small boost for positive sentiment
                        rec['recommendation_score'] = min(
                            rec['recommendation_score'] + sentiment_boost, 
                            1.0
                        )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Error enhancing with sentiment: {e}")
            return recommendations
    
    def _get_popular_restaurants(self, num_recommendations: int) -> List[Dict[str, Any]]:
        """Get popular restaurants as final fallback"""
        try:
            if not self.ratings_df.empty:
                # Calculate popularity
                popularity = self.ratings_df.groupby('restaurant_id').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                popularity.columns = ['avg_rating', 'rating_count']
                
                # Merge with restaurant details
                popular = self.restaurants_df.merge(
                    popularity, 
                    left_on='restaurant_id', 
                    right_index=True, 
                    how='left'
                )
                
                popular['avg_rating'] = popular['avg_rating'].fillna(4.0)
                popular['rating_count'] = popular['rating_count'].fillna(1)
                
                # Calculate popularity score
                popular['popularity_score'] = (
                    popular['avg_rating'] * np.log1p(popular['rating_count'])
                )
                
                top_restaurants = popular.nlargest(num_recommendations, 'popularity_score')
            else:
                # No ratings data, return first N restaurants
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
                    'recommendation_score': float(restaurant.get('popularity_score', 4.0)) / 10.0,  # Normalize
                    'method': 'popular_fallback'
                }
                
                if 'address' in restaurant:
                    rec['address'] = restaurant['address']
                if 'price_range' in restaurant:
                    rec['price_range'] = restaurant['price_range']
                    
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting popular restaurants: {e}")
            return []
