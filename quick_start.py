#!/usr/bin/env python3
"""
Simple test to run the recommendation system with LLM features
"""

import sys
import os
from pathlib import Path

def test_basic_functionality():
    """Test basic system without heavy dependencies"""
    print("🧪 Testing Basic System Functionality")
    print("=" * 40)
    
    try:
        # Test configuration
        print("📄 Testing configuration...")
        from config import Config
        print(f"✅ Config loaded")
        print(f"   LLM Enhancement: {Config.USE_LLM_ENHANCEMENT}")
        print(f"   GitHub Token: {'✅ Configured' if Config.GITHUB_TOKEN else '❌ Missing'}")
        
        # Test LLM recommender (simple version)
        print("\n🤖 Testing LLM recommender...")
        from models.llm_recommender import LLMEnhancedRecommender
        
        llm_recommender = LLMEnhancedRecommender()
        print("✅ LLM recommender initialized")
        
        # Test a simple explanation
        user_profile = {
            'preferred_cuisine': 'Italian',
            'location': 'Mumbai',
            'budget_preference': 'Medium'
        }
        
        restaurant_data = {
            'name': 'Test Restaurant',
            'cuisine': 'Italian',
            'rating': 4.5,
            'location': 'Bandra',
            'price_range': 'Medium'
        }
        
        explanation = llm_recommender.generate_recommendation_explanation(
            user_profile, restaurant_data, {}
        )
        print(f"✅ Generated explanation: {explanation[:100]}...")
        
        # Test cuisine recommendations
        available_cuisines = ['Italian', 'Chinese', 'Indian', 'Mexican']
        cuisine_rec = llm_recommender.generate_cuisine_recommendations(
            user_profile, available_cuisines
        )
        print(f"✅ Generated cuisine recommendations")
        
        print("\n🎉 Basic LLM functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    print("\n🚀 Starting Streamlit Application...")
    print("=" * 40)
    
    try:
        import subprocess
        # Run streamlit with the app
        cmd = ["streamlit", "run", "app.py", "--server.port", "8501"]
        print("🌐 Starting server at http://localhost:8501")
        print("🔧 Features available:")
        print("   ✅ Hybrid recommendations")
        print("   ✅ Sentiment analysis")
        print("   ✅ Interactive dashboard")
        print("   🤖 LLM enhancements (if GitHub token configured)")
        print("\n💡 Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

def main():
    print("🍽️  AI Recommendation System - Quick Start")
    print("=" * 50)
    
    # Test basic functionality first
    if test_basic_functionality():
        print("\n" + "=" * 50)
        choice = input("✅ Basic tests passed! Start the web application? (y/n): ").lower().strip()
        
        if choice in ['y', 'yes', '']:
            run_streamlit_app()
        else:
            print("\n📚 To manually start the app later, run:")
            print("   streamlit run app.py")
    else:
        print("\n❌ Basic tests failed. Please check the configuration.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Add your GitHub token to .env file")
        print("3. Check the LLM_INTEGRATION_GUIDE.md for setup help")

if __name__ == "__main__":
    main()
