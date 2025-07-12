#!/usr/bin/env python3
"""
Example script demonstrating LLM-enhanced recommendation features
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Demonstrate LLM features"""
    print("🤖 LLM-Enhanced Recommendation System Demo")
    print("=" * 50)
    
    try:
        # Load configuration
        from config import Config
        print(f"LLM Enhancement: {Config.USE_LLM_ENHANCEMENT}")
        print(f"GitHub Token: {'Configured' if Config.GITHUB_TOKEN else 'Not configured'}")
        
        if not Config.USE_LLM_ENHANCEMENT or not Config.GITHUB_TOKEN:
            print("\n⚠️  LLM features are disabled.")
            print("To enable them:")
            print("1. Add your GitHub token to the .env file")
            print("2. Set USE_LLM_ENHANCEMENT=true in .env")
            return
        
        # Load data
        print("\n📊 Loading data...")
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_restaurants()
        data = preprocessor.get_processed_data()
        
        # Initialize recommender
        print("🔧 Initializing recommender...")
        from models.hybrid_recommender import HybridRecommender
        recommender = HybridRecommender()
        recommender.fit(data)
        
        # Get a sample user
        sample_user = data['users'].iloc[0]
        user_id = sample_user['user_id']
        
        print(f"\n👤 Sample User Profile:")
        print(f"   User ID: {user_id}")
        print(f"   Age: {sample_user['age']}")
        print(f"   Location: {sample_user['location']}")
        print(f"   Preferred Cuisine: {sample_user['preferred_cuisine']}")
        print(f"   Budget: {sample_user['budget_preference']}")
        
        # Get enhanced recommendations
        print("\n🎯 Generating LLM-enhanced recommendations...")
        recommendations = recommender.get_hybrid_recommendations(user_id, 3)
        
        print(f"\n🍽️  Top {len(recommendations)} Recommendations:")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']} ⭐ {rec['rating']}/5")
            print(f"   Cuisine: {rec['cuisine']}")
            print(f"   Location: {rec['location']}")
            print(f"   Price: {rec['price_range']}")
            print(f"   Final Score: {rec['final_score']:.3f}")
            
            # Show LLM explanation if available
            if 'llm_explanation' in rec and rec['llm_explanation']:
                print(f"   🤖 AI Explanation: {rec['llm_explanation']}")
            else:
                print("   (No AI explanation available)")
        
        # Demonstrate cuisine recommendations
        print("\n🍜 AI Cuisine Suggestions:")
        print("-" * 30)
        cuisine_suggestions = recommender.get_cuisine_recommendations(user_id)
        print(cuisine_suggestions)
        
        # Demonstrate review summary
        if recommendations:
            sample_restaurant_id = recommendations[0]['restaurant_id']
            print(f"\n📝 AI Review Summary for '{recommendations[0]['name']}':")
            print("-" * 50)
            review_summary = recommender.get_restaurant_review_summary(sample_restaurant_id)
            print(review_summary)
        
        print("\n✅ Demo completed successfully!")
        print("💡 Run 'streamlit run app.py' to try the full interactive interface")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
