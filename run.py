import subprocess
import sys
import os
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("Creating basic .env file...")
        create_basic_env()
    else:
        print("✅ .env file found")

def create_basic_env():
    """Create a basic .env file"""
    env_content = """# GitHub Token for accessing marketplace models API
GITHUB_TOKEN=

# LLM Configuration
LLM_MODEL=gpt-4
MAX_TOKENS=1000
TEMPERATURE=0.7

# Recommendation System Settings
USE_LLM_ENHANCEMENT=false
LLM_EXPLANATION_ENABLED=false
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("📄 Basic .env file created")
    print("💡 To enable LLM features, add your GitHub token to the .env file")

def setup_environment():
    """Setup the environment and install dependencies"""
    print("🚀 Setting up AI Recommendation System...")
    
    # Check environment configuration
    check_environment()
    
    # Install requirements
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Generate sample data if it doesn't exist
    if not os.path.exists('data/restaurants.csv'):
        print("📊 Generating sample data...")
        from data.data_generator import generate_restaurant_data, generate_user_data, generate_ratings_data, generate_reviews_data
        restaurants_df = generate_restaurant_data()
        users_df = generate_user_data()
        ratings_df = generate_ratings_data(restaurants_df, users_df)
        reviews_df = generate_reviews_data(ratings_df)
        
        restaurants_df.to_csv('data/restaurants.csv', index=False)
        users_df.to_csv('data/users.csv', index=False)
        ratings_df.to_csv('data/ratings.csv', index=False)
        reviews_df.to_csv('data/reviews.csv', index=False)
        
        print("✅ Sample data generated!")
    else:
        print("✅ Sample data already exists")

def run_app():
    """Run the Streamlit application"""
    print("🍽️  Starting AI Restaurant Recommendation System...")
    print("🌐 The app will open in your browser at http://localhost:8501")
    print("\n💡 Features available:")
    print("   • Hybrid recommendation engine")
    print("   • Sentiment analysis")
    print("   • Interactive analytics dashboard")
    
    # Check if LLM is configured
    try:
        from dotenv import load_dotenv
        load_dotenv()
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            print("   • 🤖 LLM-enhanced explanations")
            print("   • 🧠 AI-powered insights")
        else:
            print("   • ⚠️  LLM features disabled (no GitHub token)")
    except:
        pass
    
    print("\n🎯 Press Ctrl+C to stop the application")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    try:
        setup_environment()
        run_app()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running: python setup.py --all")
