"""
Emotional Intelligence Engine for Restaurant Recommendations

This module provides comprehensive emotional state detection and analysis
for personalizing restaurant recommendations based on user's emotional context.
"""

import logging
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmotionalState:
    """Data class representing user's emotional state"""
    primary_emotion: str
    secondary_emotion: Optional[str]
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Union[str, float]]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalState':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ContextualFactors:
    """Environmental and contextual factors affecting mood"""
    weather: Optional[str] = None
    temperature: Optional[float] = None
    time_of_day: str = 'unknown'
    day_of_week: str = 'unknown'
    calendar_stress: float = 0.0
    social_situation: str = 'unknown'
    location_type: str = 'unknown'
    
    def to_dict(self) -> Dict:
        return asdict(self)

class EmotionDetector(ABC):
    """Abstract base class for emotion detection methods"""
    
    @abstractmethod
    def detect_emotion(self, text: str, context: Optional[ContextualFactors] = None) -> EmotionalState:
        """Detect emotion from text input"""
        pass

class VADEREmotionDetector(EmotionDetector):
    """VADER-based emotion detection with enhanced emotional mapping"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.emotion_mapping = self._load_emotion_mapping()
    
    def _load_emotion_mapping(self) -> Dict[str, Dict[str, float]]:
        """Load sentiment to emotion mapping"""
        return {
            'positive_high': {'happy': 0.8, 'excited': 0.7, 'energetic': 0.6},
            'positive_low': {'calm': 0.7, 'nostalgic': 0.5, 'romantic': 0.6},
            'negative_high': {'stressed': 0.8, 'anxious': 0.7, 'frustrated': 0.6},
            'negative_low': {'sad': 0.7, 'disappointed': 0.5, 'melancholic': 0.4},
            'neutral': {'calm': 0.5, 'neutral': 0.8, 'contemplative': 0.3}
        }
    
    def detect_emotion(self, text: str, context: Optional[ContextualFactors] = None) -> EmotionalState:
        """Detect emotion using VADER sentiment analysis"""
        if not text.strip():
            return self._default_emotional_state(context)
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Map sentiment to emotions
        if compound >= 0.5:
            emotion_category = 'positive_high'
        elif compound >= 0.1:
            emotion_category = 'positive_low'
        elif compound <= -0.5:
            emotion_category = 'negative_high'
        elif compound <= -0.1:
            emotion_category = 'negative_low'
        else:
            emotion_category = 'neutral'
        
        emotions = self.emotion_mapping[emotion_category]
        primary_emotion = max(emotions.keys(), key=lambda k: emotions[k])
        secondary_emotion = None
        
        if len(emotions) > 1:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            secondary_emotion = sorted_emotions[1][0]
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            secondary_emotion=secondary_emotion,
            intensity=abs(compound),
            confidence=emotions[primary_emotion],
            context=context.to_dict() if context else {},
            timestamp=datetime.now()
        )
    
    def _default_emotional_state(self, context: Optional[ContextualFactors]) -> EmotionalState:
        """Return default emotional state when no text is provided"""
        return EmotionalState(
            primary_emotion='neutral',
            secondary_emotion=None,
            intensity=0.5,
            confidence=0.3,
            context=context.to_dict() if context else {},
            timestamp=datetime.now()
        )

class TransformerEmotionDetector(EmotionDetector):
    """Advanced emotion detection using transformer models"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                device=-1  # Use CPU
            )
            self.available = True
            logger.info(f"Transformer emotion detector initialized with {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize transformer model: {e}")
            self.available = False
            self.fallback_detector = VADEREmotionDetector()
    
    def detect_emotion(self, text: str, context: Optional[ContextualFactors] = None) -> EmotionalState:
        """Detect emotion using transformer model"""
        if not self.available:
            return self.fallback_detector.detect_emotion(text, context)
        
        if not text.strip():
            return self._default_emotional_state(context)
        
        try:
            # Truncate text to avoid token limits
            text = text[:512] if len(text) > 512 else text
            
            results = self.classifier(text)
            
            if results:
                primary_result = results[0]
                primary_emotion = primary_result['label'].lower()
                confidence = primary_result['score']
                
                secondary_emotion = None
                if len(results) > 1:
                    secondary_emotion = results[1]['label'].lower()
                
                return EmotionalState(
                    primary_emotion=primary_emotion,
                    secondary_emotion=secondary_emotion,
                    intensity=confidence,
                    confidence=confidence,
                    context=context.to_dict() if context else {},
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Transformer emotion detection failed: {e}")
            return self.fallback_detector.detect_emotion(text, context)
        
        return self._default_emotional_state(context)
    
    def _default_emotional_state(self, context: Optional[ContextualFactors]) -> EmotionalState:
        """Return default emotional state"""
        return EmotionalState(
            primary_emotion='neutral',
            secondary_emotion=None,
            intensity=0.5,
            confidence=0.3,
            context=context.to_dict() if context else {},
            timestamp=datetime.now()
        )

class ContextAnalyzer:
    """Analyzes contextual factors that influence emotional state"""
    
    def __init__(self):
        self.weather_api_key = Config.WEATHER_API_KEY
    
    def analyze_context(self, user_location: Optional[str] = None) -> ContextualFactors:
        """Analyze current contextual factors"""
        now = datetime.now()
        
        context = ContextualFactors(
            time_of_day=self._get_time_of_day(now),
            day_of_week=now.strftime('%A').lower(),
            calendar_stress=self._estimate_calendar_stress(now),
            social_situation='unknown'
        )
        
        # Add weather data if available
        if user_location and self.weather_api_key:
            weather_data = self._get_weather_data(user_location)
            if weather_data:
                context.weather = weather_data.get('description')
                context.temperature = weather_data.get('temperature')
        
        return context
    
    def _get_time_of_day(self, dt: datetime) -> str:
        """Categorize time of day"""
        hour = dt.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _estimate_calendar_stress(self, dt: datetime) -> float:
        """Estimate stress level based on time patterns"""
        # Simple heuristic: higher stress during work hours and weekdays
        if dt.weekday() < 5:  # Weekday
            if 9 <= dt.hour <= 17:  # Work hours
                return 0.7
            elif 7 <= dt.hour <= 9 or 17 <= dt.hour <= 19:  # Commute hours
                return 0.8
        return 0.3
    
    def _get_weather_data(self, location: str) -> Optional[Dict]:
        """Fetch weather data from API"""
        if not self.weather_api_key:
            return None
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'description': data['weather'][0]['description'],
                    'temperature': data['main']['temp']
                }
        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
        
        return None

class EmotionalStateManager:
    """Manages emotional state detection, caching, and history"""
    
    def __init__(self):
        self.detector = self._initialize_detector()
        self.context_analyzer = ContextAnalyzer()
        self.cache = {}  # In-memory cache for emotional states
        self.cache_ttl = Config.EMOTIONAL_CACHE_TTL
        self._ensure_data_directory()
    
    def _initialize_detector(self) -> EmotionDetector:
        """Initialize the best available emotion detector"""
        try:
            return TransformerEmotionDetector()
        except Exception as e:
            logger.warning(f"Failed to initialize transformer detector, using VADER: {e}")
            return VADEREmotionDetector()
    
    def _ensure_data_directory(self):
        """Ensure emotional data directory exists"""
        os.makedirs(Config.EMOTIONAL_DATA_DIR, exist_ok=True)
    
    def detect_current_emotion(self, user_id: str, text_input: str = "", 
                             user_location: str = None) -> EmotionalState:
        """Detect user's current emotional state"""
        # Check cache first
        cache_key = f"{user_id}_{hash(text_input)}"
        cached_state = self._get_cached_state(cache_key)
        if cached_state:
            logger.info(f"Using cached emotional state for user {user_id}")
            return cached_state
        
        # Analyze context
        context = self.context_analyzer.analyze_context(user_location)
        
        # Detect emotion
        emotional_state = self.detector.detect_emotion(text_input, context)
        
        # Cache the result
        self._cache_state(cache_key, emotional_state)
        
        # Store in history
        self._store_emotional_history(user_id, emotional_state)
        
        logger.info(f"Detected emotion for user {user_id}: {emotional_state.primary_emotion} "
                   f"(intensity: {emotional_state.intensity:.2f})")
        
        return emotional_state
    
    def get_emotional_history(self, user_id: str, days: int = 7) -> List[EmotionalState]:
        """Get user's emotional history for the past N days"""
        history_file = os.path.join(Config.EMOTIONAL_DATA_DIR, f"user_{user_id}_history.json")
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_states = []
            
            for state_data in data:
                state = EmotionalState.from_dict(state_data)
                if state.timestamp >= cutoff_date:
                    recent_states.append(state)
            
            return sorted(recent_states, key=lambda x: x.timestamp, reverse=True)
        
        except Exception as e:
            logger.error(f"Failed to load emotional history for user {user_id}: {e}")
            return []
    
    def get_emotional_patterns(self, user_id: str) -> Dict[str, float]:
        """Analyze user's emotional patterns"""
        history = self.get_emotional_history(user_id, days=30)
        
        if not history:
            return {}
        
        # Count emotion frequencies
        emotion_counts = {}
        total_count = len(history)
        
        for state in history:
            emotion = state.primary_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Convert to percentages
        patterns = {emotion: count / total_count 
                   for emotion, count in emotion_counts.items()}
        
        return patterns
    
    def _get_cached_state(self, cache_key: str) -> Optional[EmotionalState]:
        """Get emotional state from cache if still valid"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_state(self, cache_key: str, state: EmotionalState):
        """Cache emotional state"""
        self.cache[cache_key] = (state, datetime.now())
    
    def _store_emotional_history(self, user_id: str, state: EmotionalState):
        """Store emotional state in user's history"""
        history_file = os.path.join(Config.EMOTIONAL_DATA_DIR, f"user_{user_id}_history.json")
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load existing history: {e}")
        
        # Add new state
        history.append(state.to_dict())
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(history) > 1000:
            history = history[-1000:]
        
        # Save updated history
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emotional history: {e}")

class EmotionalIntelligenceEngine:
    """Main engine for emotional intelligence in restaurant recommendations"""
    
    def __init__(self):
        self.state_manager = EmotionalStateManager()
        self.emotion_cuisine_mapping = self._load_emotion_cuisine_mapping()
        self.restaurant_mood_scores = {}
    
    def _load_emotion_cuisine_mapping(self) -> Dict[str, Dict[str, float]]:
        """Load mapping between emotions and cuisine preferences"""
        # This would ideally be loaded from a data file or learned from user behavior
        return {
            'happy': {'italian': 0.8, 'mexican': 0.7, 'thai': 0.6, 'japanese': 0.5},
            'sad': {'italian': 0.9, 'american': 0.8, 'comfort_food': 0.9, 'chinese': 0.6},
            'stressed': {'japanese': 0.8, 'vegetarian': 0.7, 'mediterranean': 0.6, 'tea_house': 0.9},
            'excited': {'mexican': 0.9, 'indian': 0.8, 'thai': 0.8, 'fusion': 0.7},
            'anxious': {'japanese': 0.9, 'vegetarian': 0.8, 'cafe': 0.8, 'quiet_dining': 0.9},
            'romantic': {'french': 0.9, 'italian': 0.8, 'fine_dining': 0.9, 'wine_bar': 0.8},
            'energetic': {'mexican': 0.8, 'korean': 0.7, 'sports_bar': 0.6, 'bbq': 0.7},
            'calm': {'japanese': 0.9, 'vegetarian': 0.8, 'cafe': 0.7, 'mediterranean': 0.6},
            'adventurous': {'fusion': 0.9, 'ethiopian': 0.8, 'korean': 0.7, 'experimental': 0.9},
            'nostalgic': {'american': 0.8, 'diner': 0.9, 'comfort_food': 0.8, 'family_style': 0.7}
        }
    
    def get_emotional_restaurant_scores(self, user_id: str, restaurants: pd.DataFrame,
                                      text_input: str = "", user_location: str = None) -> Dict[str, float]:
        """Calculate emotional compatibility scores for restaurants"""
        # Detect current emotional state
        emotional_state = self.state_manager.detect_current_emotion(
            user_id, text_input, user_location
        )
        
        scores = {}
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # Get cuisine preferences for current emotion
        cuisine_preferences = self.emotion_cuisine_mapping.get(primary_emotion, {})
        
        for _, restaurant in restaurants.iterrows():
            restaurant_id = restaurant.get('restaurant_id', restaurant.get('id'))
            cuisine = restaurant.get('cuisine', '').lower()
            
            # Base emotional score from cuisine matching
            emotional_score = cuisine_preferences.get(cuisine, 0.3)  # Default neutral score
            
            # Adjust score based on emotional intensity
            emotional_score *= intensity
            
            # Add restaurant-specific mood attributes if available
            mood_score = self._calculate_restaurant_mood_score(restaurant, emotional_state)
            
            # Combine scores
            final_score = (emotional_score * 0.7) + (mood_score * 0.3)
            scores[str(restaurant_id)] = final_score
        
        return scores
    
    def _calculate_restaurant_mood_score(self, restaurant: pd.Series, 
                                       emotional_state: EmotionalState) -> float:
        """Calculate restaurant's mood compatibility score"""
        # This would use restaurant atmosphere data if available
        # For now, return a neutral score
        return 0.5
    
    def get_emotional_explanation(self, emotional_state: EmotionalState, 
                                restaurant_data: Dict) -> str:
        """Generate explanation for emotional-based recommendation"""
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        restaurant_name = restaurant_data.get('name', 'this restaurant')
        cuisine = restaurant_data.get('cuisine', 'cuisine')
        
        explanations = {
            'happy': f"You're feeling {emotion}! {restaurant_name} with its {cuisine} cuisine "
                    f"is perfect for celebrating and enjoying good vibes.",
            'sad': f"When feeling {emotion}, comfort food can help. {restaurant_name} offers "
                   f"the kind of {cuisine} comfort that might lift your spirits.",
            'stressed': f"You seem {emotion}. {restaurant_name} provides a calming {cuisine} "
                       f"experience that can help you unwind and relax.",
            'excited': f"Your {emotion} energy matches perfectly with {restaurant_name}'s "
                      f"vibrant {cuisine} atmosphere!",
            'anxious': f"For your {emotion} mood, {restaurant_name} offers a peaceful "
                      f"{cuisine} environment that promotes calm.",
            'romantic': f"Perfect for a {emotion} evening! {restaurant_name}'s {cuisine} "
                       f"setting creates an intimate dining experience.",
            'energetic': f"Your {emotion} vibe aligns with {restaurant_name}'s lively "
                        f"{cuisine} atmosphere!",
            'calm': f"In your {emotion} state, {restaurant_name} provides a serene "
                   f"{cuisine} experience that matches your peaceful mood.",
            'adventurous': f"Feeling {emotion}? {restaurant_name}'s {cuisine} menu offers "
                          f"exciting flavors for your culinary exploration!",
            'nostalgic': f"Your {emotion} mood calls for familiar flavors. {restaurant_name}'s "
                        f"{cuisine} dishes evoke wonderful memories."
        }
        
        base_explanation = explanations.get(emotion, 
            f"Based on your {emotion} mood, {restaurant_name} seems like a great choice!")
        
        if intensity > 0.7:
            base_explanation += f" Your strong {emotion} feeling makes this an especially good match!"
        
        return base_explanation
    
    def detect_emotion(self, text_input: str, user_location: str = None) -> Dict[str, Union[str, float]]:
        """Wrapper method for detecting emotions from text input"""
        # Use a dummy user_id for standalone emotion detection
        emotional_state = self.state_manager.detect_current_emotion(
            "anonymous", text_input, user_location
        )
        
        return {
            'primary_emotion': emotional_state.primary_emotion,
            'secondary_emotion': emotional_state.secondary_emotion,
            'intensity': emotional_state.intensity,
            'confidence': emotional_state.confidence,
            'context': emotional_state.context
        }
