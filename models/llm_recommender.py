import os
import json
import requests
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEnhancedRecommender:
    """Multi-provider LLM-enhanced recommender supporting GitHub, Hugging Face, and OpenAI"""
    
    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        self.model_name = Config.LLM_MODEL
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE
        
        # Initialize based on provider
        if self.provider == 'huggingface':
            self._init_huggingface()
        elif self.provider == 'github':
            self._init_github()
        elif self.provider == 'openai':
            self._init_openai()
        else:
            logger.warning(f"Unknown provider: {self.provider}, falling back to simple responses")
            self.llm_available = False
    
    def _init_huggingface(self):
        """Initialize Hugging Face API"""
        self.hf_token = Config.HUGGINGFACE_TOKEN
        if not self.hf_token or self.hf_token.strip() == 'your_huggingface_token_here':
            logger.warning("Hugging Face token not configured")
            self.llm_available = False
            return
        
        try:
            # Try using requests for Hugging Face Inference API first (lighter)
            self.hf_headers = {
                "Authorization": f"Bearer {self.hf_token.strip()}",
                "Content-Type": "application/json"
            }
            
            # Use a good model for text generation
            if 'gpt' in self.model_name.lower() or self.model_name == 'gpt-4':
                self.model_name = "microsoft/DialoGPT-medium"
            elif not self.model_name:
                self.model_name = "microsoft/DialoGPT-medium"
            
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            
            # Test the API with better error handling
            try:
                test_response = requests.post(
                    self.hf_api_url,
                    headers=self.hf_headers,
                    json={"inputs": "Test", "parameters": {"max_new_tokens": 10}},
                    timeout=10
                )
                
                if test_response.status_code == 200:
                    self.llm_available = True
                    logger.info(f"Hugging Face API with model '{self.model_name}' initialized successfully")
                elif test_response.status_code == 503:
                    # Model is loading, still available but may need retries
                    self.llm_available = True
                    logger.info(f"Hugging Face model '{self.model_name}' is loading, will retry requests")
                else:
                    logger.warning(f"Hugging Face API test failed: {test_response.status_code} - {test_response.text[:100]}")
                    self.llm_available = False
            except requests.exceptions.Timeout:
                # Still mark as available, timeout might be temporary
                logger.warning("Hugging Face API timeout during test, but marking as available")
                self.llm_available = True
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to Hugging Face API")
                self.llm_available = False
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face API: {e}")
            self.llm_available = False
    
    def _init_github(self):
        """Initialize GitHub marketplace models"""
        self.github_token = Config.GITHUB_TOKEN
        if not self.github_token or self.github_token.strip() == 'your_github_token_here':
            logger.warning("GitHub token not configured")
            self.llm_available = False
            return
        
        # For GitHub models, we'll use direct API calls
        self.github_headers = {
            "Authorization": f"Bearer {self.github_token.strip()}",
            "Content-Type": "application/json"
        }
        
        self.llm_available = True
        logger.info("GitHub marketplace API initialized")
    
    def _init_openai(self):
        """Initialize OpenAI API"""
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                logger.warning("OpenAI API key not configured")
                self.llm_available = False
                return
            
            # Test if the API key is valid by trying an import
            from openai import OpenAI
            self.llm_available = True
            logger.info("OpenAI API initialized")
            
        except ImportError:
            logger.error("OpenAI library not installed")
            self.llm_available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.llm_available = False
    
    def _call_llm(self, prompt, max_length=200):
        """Call the appropriate LLM based on provider"""
        if not self.llm_available:
            return self._get_fallback_response(prompt)
        
        try:
            if self.provider == 'huggingface':
                return self._call_huggingface(prompt, max_length)
            elif self.provider == 'github':
                return self._call_github_api(prompt, max_length)
            elif self.provider == 'openai':
                return self._call_openai(prompt, max_length)
            else:
                return self._get_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._get_fallback_response(prompt)
    
    def _call_huggingface(self, prompt, max_length=200):
        """Call Hugging Face Inference API with improved error handling"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Truncate prompt to avoid token limits
                if len(prompt) > 500:
                    prompt = prompt[:500] + "..."
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": min(max_length, 150),
                        "temperature": self.temperature,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(
                    self.hf_api_url,
                    headers=self.hf_headers,
                    json=payload,
                    timeout=20  # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                        return generated_text if generated_text else self._get_fallback_response(prompt)
                    else:
                        return self._get_fallback_response(prompt)
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt < max_retries - 1:
                        logger.info(f"Model loading, retrying in 3 seconds (attempt {attempt + 1})")
                        import time
                        time.sleep(3)
                        continue
                    else:
                        logger.warning("Model still loading after retries, using fallback")
                        return self._get_fallback_response(prompt)
                else:
                    logger.warning(f"Hugging Face API error: {response.status_code} - {response.text[:100]}")
                    return self._get_fallback_response(prompt)
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"HF API timeout, retrying (attempt {attempt + 1})")
                    continue
                else:
                    logger.error("HF API timeout after retries")
                    return self._get_fallback_response(prompt)
            except Exception as e:
                logger.error(f"Hugging Face API error: {e}")
                return self._get_fallback_response(prompt)
        
        return self._get_fallback_response(prompt)
    
    def _call_github_api(self, prompt, max_length=200):
        """Call GitHub marketplace models API"""
        try:
            # This is a placeholder for GitHub API integration
            # You would implement the actual GitHub marketplace API calls here
            logger.info("GitHub API call - using fallback for now")
            return self._get_fallback_response(prompt)
            
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
            return self._get_fallback_response(prompt)
    
    def _call_openai(self, prompt, max_length=200):
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(max_length, self.max_tokens),
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt):
        """Provide intelligent fallback responses based on prompt content"""
        prompt_lower = prompt.lower()
        
        if "why" in prompt_lower and "recommend" in prompt_lower:
            return "This restaurant is recommended based on your taste preferences, location, and positive customer reviews from similar users."
        elif "cuisine" in prompt_lower and "suggest" in prompt_lower:
            return "Based on your preferences, you might enjoy exploring similar cuisines with good ratings and positive reviews."
        elif "review" in prompt_lower and "summary" in prompt_lower:
            return "Customer reviews are generally positive, highlighting good food quality and service. Some reviews mention areas for improvement."
        else:
            return "This recommendation is based on advanced AI analysis of your preferences and similar user patterns."
    
    def generate_recommendation_explanation(self, user_profile, restaurant_data, recommendation_scores):
        """Generate detailed explanation for recommendations using LLM"""
        
        prompt = f"""
        Explain why this restaurant is recommended for this user in 2-3 sentences:
        
        User Profile:
        - Age: {user_profile.get('age', 'N/A')}
        - Location: {user_profile.get('location', 'N/A')}
        - Preferred Cuisine: {user_profile.get('preferred_cuisine', 'N/A')}
        - Budget: {user_profile.get('budget_preference', 'N/A')}
        
        Restaurant:
        - Name: {restaurant_data.get('name', 'N/A')}
        - Cuisine: {restaurant_data.get('cuisine', 'N/A')}
        - Rating: {restaurant_data.get('rating', 'N/A')}/5
        - Location: {restaurant_data.get('location', 'N/A')}
        - Price: {restaurant_data.get('price_range', 'N/A')}
        
        Recommendation Score: {recommendation_scores.get('final', 0):.3f}
        
        Provide a personalized explanation:
        """
        
        return self._call_llm(prompt, 150)
    
    def generate_cuisine_recommendations(self, user_preferences, available_cuisines):
        """Generate cuisine recommendations based on user preferences using LLM"""
        
        prompt = f"""
        Suggest 3 cuisines for this user:
        
        User Preferences:
        - Preferred Cuisine: {user_preferences.get('preferred_cuisine', 'N/A')}
        - Location: {user_preferences.get('location', 'N/A')}
        - Budget: {user_preferences.get('budget_preference', 'N/A')}
        
        Available: {', '.join(available_cuisines[:10])}
        
        Provide 3 suggestions with brief reasons:
        """
        
        return self._call_llm(prompt, 200)
    
    def generate_restaurant_review_summary(self, reviews_data):
        """Generate a summary of restaurant reviews using LLM"""
        
        if not reviews_data or len(reviews_data) == 0:
            return "No reviews available for this restaurant."
        
        # Limit reviews to avoid token limits
        limited_reviews = reviews_data[:5] if len(reviews_data) > 5 else reviews_data
        reviews_text = "\n".join([f"- Rating: {review.get('rating', 'N/A')}/5, Review: {str(review.get('review_text', ''))[:100]}" 
                                 for review in limited_reviews])
        
        prompt = f"""
        Summarize these restaurant reviews in 2-3 sentences:
        
        Reviews:
        {reviews_text}
        
        Provide a balanced summary highlighting positives and concerns:
        """
        
        return self._call_llm(prompt, 150)
    
    def generate_personalized_dining_suggestions(self, user_profile, time_of_day, weather=None):
        """Generate personalized dining suggestions based on context"""
        
        prompt = f"""
        Suggest dining options for this user:
        
        User: {user_profile.get('preferred_cuisine', 'Any')} cuisine lover, {user_profile.get('budget_preference', 'medium')} budget
        Time: {time_of_day}
        Weather: {weather or 'Unknown'}
        
        Provide 2-3 specific suggestions:
        """
        
        return self._call_llm(prompt, 150)
    
    def enhance_recommendations_with_llm(self, recommendations, user_profile):
        """Enhance existing recommendations with LLM-generated explanations"""
        
        enhanced_recommendations = []
        
        for rec in recommendations:
            try:
                # Generate explanation for this recommendation
                explanation = self.generate_recommendation_explanation(
                    user_profile,
                    rec,
                    {
                        'collaborative': rec.get('collaborative_score', 0),
                        'content': rec.get('content_score', 0),
                        'sentiment': rec.get('sentiment_score', 0),
                        'final': rec.get('final_score', 0)
                    }
                )
                
                # Add LLM explanation to recommendation
                enhanced_rec = rec.copy()
                enhanced_rec['llm_explanation'] = explanation
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                logger.error(f"Error enhancing recommendation for {rec.get('name', 'Unknown')}: {e}")
                # Add without enhancement if LLM fails
                enhanced_recommendations.append(rec)
        
        return enhanced_recommendations
