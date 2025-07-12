import os
import json
import requests
from config import Config
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEnhancedRecommender:
    """LLM-enhanced recommender with multiple fallback options"""
    
    def __init__(self):
        self.github_token = Config.GITHUB_TOKEN
        self.llm_available = False
        
        if not self.github_token:
            logger.warning("GitHub token not found - using fallback responses")
            return
        
        # Test if we can use the LLM
        try:
            test_response = self._call_github_models_api("Test connection")
            if test_response and "error" not in test_response.lower():
                self.llm_available = True
                logger.info("GitHub Models API connection successful")
            else:
                logger.warning("GitHub Models API test failed - using fallback responses")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e} - using fallback responses")
    
    def _call_github_models_api(self, prompt):
        """Call GitHub Models API directly"""
        try:
            # GitHub Models API endpoint
            url = "https://models.inference.ai.azure.com/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": Config.LLM_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful restaurant recommendation assistant."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": Config.MAX_TOKENS,
                "temperature": Config.TEMPERATURE
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"GitHub API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None
    
    def _call_llm(self, prompt):
        """Call LLM with fallback to template responses"""
        if self.llm_available:
            try:
                response = self._call_github_models_api(prompt)
                if response:
                    return response.strip()
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        # Use fallback template responses
        return self._get_template_response(prompt)
    
    def _get_template_response(self, prompt):
        """Generate template-based responses when LLM is not available"""
        prompt_lower = prompt.lower()
        
        if "explanation" in prompt_lower:
            explanations = [
                "This restaurant is recommended because it matches your cuisine preferences and has excellent customer reviews.",
                "Based on your dining history and preferences, this restaurant offers the perfect combination of quality and value.",
                "This recommendation aligns with your taste profile and is highly rated by customers with similar preferences.",
                "The restaurant's cuisine style, location, and customer feedback make it an ideal match for your preferences.",
                "This establishment combines your preferred cuisine type with outstanding customer satisfaction ratings."
            ]
            return random.choice(explanations)
            
        elif "cuisine" in prompt_lower:
            return """Based on your preferences, here are some cuisine recommendations:
1. Italian: Great for authentic flavors and diverse pasta options
2. Asian Fusion: Perfect blend of traditional and modern cooking styles  
3. Mediterranean: Healthy options with fresh ingredients and bold flavors"""
            
        elif "review" in prompt_lower:
            summaries = [
                "Customers consistently praise the authentic flavors and generous portions, with many highlighting excellent service.",
                "Reviews indicate high satisfaction with food quality and ambiance, though some mention occasional wait times during peak hours.",
                "Overall positive feedback focuses on fresh ingredients and friendly staff, with consistent quality across visits.",
                "Customer reviews emphasize great value for money and authentic cuisine, with particularly strong ratings for service quality."
            ]
            return random.choice(summaries)
            
        elif "dining" in prompt_lower:
            return """Consider these dining options based on current context:
• Light lunch options with fresh, healthy ingredients
• Comfort food that matches the current weather and time
• Local specialties that offer authentic cultural experiences"""
            
        else:
            return "AI analysis suggests this is a good match based on your preferences and dining patterns."
    
    def generate_recommendation_explanation(self, user_profile, restaurant_data, recommendation_scores):
        """Generate detailed explanation for recommendations"""
        
        prompt = f"""
        Explain why {restaurant_data.get('name', 'this restaurant')} is recommended for a user who:
        - Prefers {user_profile.get('preferred_cuisine', 'various')} cuisine
        - Lives in {user_profile.get('location', 'the area')}
        - Has a {user_profile.get('budget_preference', 'moderate')} budget
        
        The restaurant offers {restaurant_data.get('cuisine', 'great')} cuisine, 
        is located in {restaurant_data.get('location', 'a convenient area')}, 
        and has a {restaurant_data.get('rating', 'good')} star rating.
        
        Provide a friendly, personalized explanation in 2-3 sentences.
        """
        
        return self._call_llm(prompt)
    
    def generate_cuisine_recommendations(self, user_profile, available_cuisines):
        """Generate cuisine recommendations based on user preferences"""
        
        preferred = user_profile.get('preferred_cuisine', 'Various')
        location = user_profile.get('location', 'your area')
        budget = user_profile.get('budget_preference', 'moderate')
        
        prompt = f"""
        A user in {location} who prefers {preferred} cuisine and has a {budget} budget
        is looking for new cuisine recommendations. Available options: {', '.join(available_cuisines[:10])}
        
        Suggest 3 cuisines they might enjoy and briefly explain why each would appeal to them.
        Format as a numbered list with brief explanations.
        """
        
        return self._call_llm(prompt)
    
    def generate_restaurant_review_summary(self, reviews_data):
        """Generate a summary of restaurant reviews"""
        
        if not reviews_data or len(reviews_data) == 0:
            return "No reviews available for this restaurant."
        
        # Analyze review sentiment and common themes
        total_reviews = len(reviews_data)
        avg_rating = sum(r.get('rating', 3) for r in reviews_data) / total_reviews if reviews_data else 3
        
        # Sample some review text for analysis
        sample_reviews = reviews_data[:5] if len(reviews_data) > 5 else reviews_data
        review_texts = [r.get('review_text', '')[:100] for r in sample_reviews if r.get('review_text')]
        
        prompt = f"""
        Summarize the customer experience for a restaurant based on {total_reviews} reviews 
        with an average rating of {avg_rating:.1f}/5. 
        
        Sample review excerpts: {'; '.join(review_texts[:3])}
        
        Provide a balanced 2-3 sentence summary highlighting the main positive aspects 
        and any areas for improvement mentioned by customers.
        """
        
        return self._call_llm(prompt)
    
    def generate_personalized_dining_suggestions(self, user_profile, time_of_day, weather=None):
        """Generate personalized dining suggestions based on context"""
        
        cuisine = user_profile.get('preferred_cuisine', 'various cuisines')
        budget = user_profile.get('budget_preference', 'moderate')
        
        prompt = f"""
        Suggest dining options for someone who enjoys {cuisine} with a {budget} budget.
        Current context: {time_of_day} time
        {f'Weather: {weather}' if weather else ''}
        
        Provide 2-3 specific dining suggestions that would be perfect for this time and context.
        """
        
        return self._call_llm(prompt)
    
    def _format_restaurant_data(self, restaurant_data):
        """Format restaurant data for prompts"""
        return f"""
        Restaurant: {restaurant_data.get('name', 'N/A')}
        Cuisine: {restaurant_data.get('cuisine', 'N/A')}
        Rating: {restaurant_data.get('rating', 'N/A')}/5
        Location: {restaurant_data.get('location', 'N/A')}
        Price Range: {restaurant_data.get('price_range', 'N/A')}
        """
    
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
