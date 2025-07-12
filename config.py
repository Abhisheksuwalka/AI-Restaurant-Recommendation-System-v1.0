import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Data paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    EMOTIONAL_DATA_DIR = 'data/emotional'
    
    # Model parameters
    N_RECOMMENDATIONS = 10
    SIMILARITY_THRESHOLD = 0.1
    
    # Sentiment analysis
    SENTIMENT_THRESHOLD = 0.1
    
    # UI settings
    PAGE_TITLE = "AI Restaurant Recommendation System"
    PAGE_ICON = "🍽️"
    
    # LLM Configuration
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    LLM_MODEL = os.getenv('LLM_MODEL', 'microsoft/DialoGPT-medium')
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'huggingface')  # github, huggingface, openai
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    USE_LLM_ENHANCEMENT = os.getenv('USE_LLM_ENHANCEMENT', 'true').lower() == 'true'
    LLM_EXPLANATION_ENABLED = os.getenv('LLM_EXPLANATION_ENABLED', 'true').lower() == 'true'
    
    # Emotional State Configuration
    USE_EMOTIONAL_RECOMMENDATIONS = os.getenv('USE_EMOTIONAL_RECOMMENDATIONS', 'true').lower() == 'true'
    EMOTIONAL_CACHE_TTL = int(os.getenv('EMOTIONAL_CACHE_TTL', '3600'))  # 1 hour
    EMOTIONAL_WEIGHT = float(os.getenv('EMOTIONAL_WEIGHT', '0.3'))  # 30% weight in final score
    
    # External API Configuration
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    GOOGLE_CALENDAR_CREDENTIALS = os.getenv('GOOGLE_CALENDAR_CREDENTIALS')
    
    # Emotional States Configuration
    EMOTIONAL_STATES = [
        'happy', 'sad', 'stressed', 'excited', 'anxious', 
        'romantic', 'energetic', 'calm', 'adventurous', 'nostalgic'
    ]
    
    # Restaurant Mood Attributes
    RESTAURANT_MOOD_ATTRIBUTES = [
        'comfort_level', 'energy_level', 'social_intimacy', 
        'adventure_factor', 'stress_relief', 'romance_factor'
    ]
