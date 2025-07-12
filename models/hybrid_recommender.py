import pandas as pd
import numpy as np
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based_filtering import ContentBasedFiltering
from models.sentiment_analyzer import SentimentAnalyzer
from models.llm_recommender import LLMEnhancedRecommender
from models.emotional_intelligence import EmotionalIntelligenceEngine
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self):
        self.collaborative_model = CollaborativeFiltering()
        self.content_model = ContentBasedFiltering()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_scores = {}
        
        # Initialize LLM enhancer if enabled
        self.llm_enhancer = None
        if Config.USE_LLM_ENHANCEMENT:
            try:
                self.llm_enhancer = LLMEnhancedRecommender()
                logger.info("LLM enhancement enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM enhancer: {e}")
                self.llm_enhancer = None
        
        # Initialize Emotional Intelligence Engine if enabled
        self.emotional_engine = None
        if Config.USE_EMOTIONAL_RECOMMENDATIONS:
            try:
                self.emotional_engine = EmotionalIntelligenceEngine()
                logger.info("Emotional intelligence enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize emotional intelligence: {e}")
                self.emotional_engine = None
        
    def fit(self, data):
        """Train all models"""
        print("Training collaborative filtering model...")
        self.collaborative_model.fit(data['user_item_matrix'])
        
        print("Training content-based filtering model...")
        self.content_model.fit(data['restaurants'], data['restaurant_features'])
        
        print("Analyzing sentiment...")
        self.sentiment_results = self.sentiment_analyzer.analyze_reviews_batch(data['reviews'])
        
        # Calculate sentiment scores for each restaurant
        for restaurant_id in data['restaurants']['restaurant_id']:
            sentiment_info = self.sentiment_analyzer.get_restaurant_sentiment_score(
                self.sentiment_results, restaurant_id
            )
            self.sentiment_scores[restaurant_id] = sentiment_info
        
        self.data = data
        print("Hybrid recommender trained successfully!")
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10, 
                                 collaborative_weight=0.3, content_weight=0.2, 
                                 sentiment_weight=0.2, emotional_weight=0.3,
                                 user_text_input="", user_location=None):
        """Get hybrid recommendations combining all approaches including emotional intelligence"""
        
        # Get collaborative filtering recommendations
        collab_recs = self.collaborative_model.get_user_recommendations(user_id, n_recommendations*2)
        
        # Get content-based recommendations (using user's historical preferences)
        user_ratings = self.data['ratings'][self.data['ratings']['user_id'] == user_id]
        if len(user_ratings) > 0:
            # Get restaurants user has rated highly
            high_rated = user_ratings[user_ratings['rating'] >= 4]['restaurant_id'].tolist()
            content_recs = []
            for restaurant_id in high_rated[:3]:  # Use top 3 liked restaurants
                similar_restaurants = self.content_model.get_restaurant_recommendations(
                    restaurant_id, n_recommendations//3
                )
                content_recs.extend(similar_restaurants)
        else:
            content_recs = []
        
        # Get emotional compatibility scores if available
        emotional_scores = {}
        if self.emotional_engine and Config.USE_EMOTIONAL_RECOMMENDATIONS:
            try:
                emotional_scores = self.emotional_engine.get_emotional_restaurant_scores(
                    user_id, self.data['restaurants'], user_text_input, user_location
                )
                logger.info(f"Generated emotional scores for {len(emotional_scores)} restaurants")
            except Exception as e:
                logger.error(f"Error generating emotional scores: {e}")
                emotional_scores = {}
        
        # Combine and score recommendations
        all_recommendations = {}
        
        # Add collaborative filtering scores
        for restaurant_id, score in collab_recs:
            if restaurant_id not in all_recommendations:
                all_recommendations[restaurant_id] = {'scores': {}, 'info': {}}
            all_recommendations[restaurant_id]['scores']['collaborative'] = score
        
        # Add content-based scores
        for rec in content_recs:
            restaurant_id = rec['restaurant_id']
            if restaurant_id not in all_recommendations:
                all_recommendations[restaurant_id] = {'scores': {}, 'info': {}}
            all_recommendations[restaurant_id]['scores']['content'] = rec['similarity_score']
            all_recommendations[restaurant_id]['info'] = rec
        
        # Ensure we have enough restaurants to recommend
        all_restaurant_ids = self.data['restaurants']['restaurant_id'].tolist()
        for restaurant_id in all_restaurant_ids:
            if restaurant_id not in all_recommendations:
                all_recommendations[restaurant_id] = {'scores': {}, 'info': {}}
        
        # Add sentiment scores and calculate final scores
        final_recommendations = []
        for restaurant_id, data in all_recommendations.items():
            scores = data['scores']
            
            # Get sentiment score
            sentiment_info = self.sentiment_scores.get(restaurant_id, {'avg_sentiment': 0.0})
            # Handle case where sentiment_info might be a float instead of dict
            if isinstance(sentiment_info, (int, float)):
                sentiment_score = max(0, sentiment_info + 1) / 2  # Normalize to 0-1
            else:
                sentiment_score = max(0, sentiment_info.get('avg_sentiment', 0.0) + 1) / 2  # Normalize to 0-1
            
            # Get emotional score
            emotional_score = emotional_scores.get(str(restaurant_id), 0.5)  # Default neutral score
            
            # Calculate weighted final score
            final_score = (
                scores.get('collaborative', 0.3) * collaborative_weight +
                scores.get('content', 0.3) * content_weight +
                sentiment_score * sentiment_weight +
                emotional_score * emotional_weight
            )
            
            # Get restaurant info
            restaurant_info = self.data['restaurants'][
                self.data['restaurants']['restaurant_id'] == restaurant_id
            ].iloc[0]
            
            rec_item = {
                'restaurant_id': restaurant_id,
                'name': restaurant_info['name'],
                'cuisine': restaurant_info['cuisine'],
                'rating': restaurant_info['rating'],
                'location': restaurant_info['location'],
                'price_range': restaurant_info['price_range'],
                'final_score': final_score,
                'sentiment_score': sentiment_score,
                'sentiment_info': sentiment_info,
                'collaborative_score': scores.get('collaborative', 0),
                'content_score': scores.get('content', 0),
                'emotional_score': emotional_score
            }
            
            # Add emotional explanation if available
            if self.emotional_engine and emotional_score > 0.5:
                try:
                    emotional_state = self.emotional_engine.state_manager.detect_current_emotion(
                        user_id, user_text_input, user_location
                    )
                    emotional_explanation = self.emotional_engine.get_emotional_explanation(
                        emotional_state, restaurant_info.to_dict()
                    )
                    rec_item['emotional_explanation'] = emotional_explanation
                except Exception as e:
                    logger.error(f"Error generating emotional explanation: {e}")
            
            final_recommendations.append(rec_item)
        
        # Sort by final score and return top N
        final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        top_recommendations = final_recommendations[:n_recommendations]
        
        # Enhance with LLM if available
        if self.llm_enhancer and Config.LLM_EXPLANATION_ENABLED:
            try:
                # Get user profile
                user_profile = self.data['users'][self.data['users']['user_id'] == user_id].iloc[0].to_dict()
                top_recommendations = self.llm_enhancer.enhance_recommendations_with_llm(
                    top_recommendations, user_profile
                )
                logger.info(f"Enhanced {len(top_recommendations)} recommendations with LLM")
            except Exception as e:
                logger.error(f"Error enhancing recommendations with LLM: {e}")
        
        return top_recommendations

    def get_cuisine_recommendations(self, user_id):
        """Get LLM-powered cuisine recommendations"""
        if not self.llm_enhancer:
            return "LLM enhancement not available"
            
        try:
            user_profile = self.data['users'][self.data['users']['user_id'] == user_id].iloc[0].to_dict()
            available_cuisines = self.data['restaurants']['cuisine'].unique().tolist()
            return self.llm_enhancer.generate_cuisine_recommendations(user_profile, available_cuisines)
        except Exception as e:
            logger.error(f"Error generating cuisine recommendations: {e}")
            return "Unable to generate cuisine recommendations"
    
    def get_restaurant_review_summary(self, restaurant_id):
        """Get LLM-generated review summary for a restaurant"""
        if not self.llm_enhancer:
            return "LLM enhancement not available"
            
        try:
            restaurant_reviews = self.data['reviews'][
                self.data['reviews']['restaurant_id'] == restaurant_id
            ].to_dict('records')
            return self.llm_enhancer.generate_restaurant_review_summary(restaurant_reviews)
        except Exception as e:
            logger.error(f"Error generating review summary: {e}")
            return "Unable to generate review summary"
    
    def get_emotional_recommendations(self, user_id, user_text_input="", user_location=None, n_recommendations=10):
        """Get recommendations based purely on emotional state"""
        if not self.emotional_engine:
            return self.get_hybrid_recommendations(user_id, n_recommendations)
        
        try:
            # Get emotional scores for all restaurants
            emotional_scores = self.emotional_engine.get_emotional_restaurant_scores(
                user_id, self.data['restaurants'], user_text_input, user_location
            )
            
            # Create recommendations based on emotional scores
            emotional_recommendations = []
            for _, restaurant in self.data['restaurants'].iterrows():
                restaurant_id = restaurant['restaurant_id']
                emotional_score = emotional_scores.get(str(restaurant_id), 0.5)
                
                # Get sentiment score for additional context
                sentiment_info = self.sentiment_scores.get(restaurant_id, {'avg_sentiment': 0.0})
                # Handle case where sentiment_info might be a float instead of dict
                if isinstance(sentiment_info, (int, float)):
                    sentiment_score = max(0, sentiment_info + 1) / 2
                else:
                    sentiment_score = max(0, sentiment_info.get('avg_sentiment', 0.0) + 1) / 2
                
                # Combine emotional and sentiment scores
                final_score = (emotional_score * 0.7) + (sentiment_score * 0.3)
                
                rec_item = {
                    'restaurant_id': restaurant_id,
                    'name': restaurant['name'],
                    'cuisine': restaurant['cuisine'],
                    'rating': restaurant['rating'],
                    'location': restaurant['location'],
                    'price_range': restaurant['price_range'],
                    'final_score': final_score,
                    'emotional_score': emotional_score,
                    'sentiment_score': sentiment_score
                }
                
                # Add emotional explanation
                try:
                    emotional_state = self.emotional_engine.state_manager.detect_current_emotion(
                        user_id, user_text_input, user_location
                    )
                    emotional_explanation = self.emotional_engine.get_emotional_explanation(
                        emotional_state, restaurant.to_dict()
                    )
                    rec_item['emotional_explanation'] = emotional_explanation
                    rec_item['detected_emotion'] = emotional_state.primary_emotion
                    rec_item['emotion_intensity'] = emotional_state.intensity
                except Exception as e:
                    logger.error(f"Error generating emotional explanation: {e}")
                
                emotional_recommendations.append(rec_item)
            
            # Sort by emotional compatibility
            emotional_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            return emotional_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting emotional recommendations: {e}")
            return self.get_hybrid_recommendations(user_id, n_recommendations)
    
    def get_user_emotional_insights(self, user_id):
        """Get insights about user's emotional patterns"""
        if not self.emotional_engine:
            return {"status": "Emotional intelligence not available"}
        
        try:
            # Get emotional history
            emotional_history = self.emotional_engine.state_manager.get_emotional_history(user_id)
            
            # Get emotional patterns
            emotional_patterns = self.emotional_engine.state_manager.get_emotional_patterns(user_id)
            
            insights = {
                "emotional_history_count": len(emotional_history),
                "emotional_patterns": emotional_patterns,
                "most_frequent_emotion": max(emotional_patterns.keys(), key=emotional_patterns.get) if emotional_patterns else "Unknown",
                "recent_emotions": [state.primary_emotion for state in emotional_history[:5]]
            }
            
            if emotional_history:
                insights["last_emotion_detected"] = emotional_history[0].primary_emotion
                insights["last_emotion_timestamp"] = emotional_history[0].timestamp.isoformat()
                insights["last_emotion_intensity"] = emotional_history[0].intensity
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting emotional insights: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def get_mood_based_cuisine_suggestions(self, user_id, user_text_input="", user_location=None):
        """Get cuisine suggestions based on current mood"""
        if not self.emotional_engine:
            return "Emotional intelligence not available"
        
        try:
            # Detect current emotional state
            emotional_state = self.emotional_engine.state_manager.detect_current_emotion(
                user_id, user_text_input, user_location
            )
            
            # Get cuisine recommendations for this emotion
            primary_emotion = emotional_state.primary_emotion
            cuisine_preferences = self.emotional_engine.emotion_cuisine_mapping.get(primary_emotion, {})
            
            # Get available cuisines from restaurant data
            available_cuisines = self.data['restaurants']['cuisine'].unique().tolist()
            
            # Match and score cuisines
            mood_cuisines = []
            for cuisine in available_cuisines:
                score = cuisine_preferences.get(cuisine.lower(), 0.3)
                mood_cuisines.append({
                    'cuisine': cuisine,
                    'emotional_compatibility': score,
                    'reason': f"Great choice when feeling {primary_emotion}"
                })
            
            # Sort by compatibility
            mood_cuisines.sort(key=lambda x: x['emotional_compatibility'], reverse=True)
            
            result = {
                "detected_emotion": primary_emotion,
                "emotion_intensity": emotional_state.intensity,
                "recommended_cuisines": mood_cuisines[:5],
                "explanation": f"Based on your {primary_emotion} mood, here are some cuisine suggestions that might appeal to you."
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting mood-based cuisine suggestions: {e}")
            return f"Error generating suggestions: {str(e)}"
