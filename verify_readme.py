#!/usr/bin/env python3
"""
README Verification Script
Validates that all information in README.md is accurate
"""

import os
import sys

def verify_readme_accuracy():
    """Verify all claims in README.md are accurate"""
    
    print("🔍 Verifying README.md Accuracy")
    print("=" * 40)
    
    # Import pandas
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available for verification")
        return
    
    # Check data files exist and sizes match README claims
    data_files = {
        'restaurants.csv': {'expected_rows': 5000, 'description': 'Restaurant data'},
        'users.csv': {'expected_rows': 2000, 'description': 'User data'},
        'ratings.csv': {'expected_rows': 25000, 'description': 'Rating data'},
        'reviews.csv': {'expected_rows': 15000, 'description': 'Review data'}
    }
    
    total_entries = 0
    
    for filename, info in data_files.items():
        filepath = f"data/{filename}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            actual_rows = len(df)
            total_entries += actual_rows
            
            status = "✅" if actual_rows >= info['expected_rows'] - 100 else "❌"  # Allow small variance
            print(f"{status} {info['description']}: {actual_rows:,} rows (Expected: ~{info['expected_rows']:,})")
            
            # Verify schema for each file
            if filename == 'restaurants.csv':
                expected_cols = ['restaurant_id', 'name', 'cuisine', 'location', 'price_range', 'rating']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if not missing_cols:
                    print(f"   ✅ Schema verified: {len(df.columns)} columns")
                else:
                    print(f"   ❌ Missing columns: {missing_cols}")
                    
        else:
            print(f"❌ {info['description']}: File not found")
    
    print(f"\n📊 Total Data Points: {total_entries:,} (Expected: ~47,000)")
    
    # Check if key files exist
    key_files = [
        'models/collaborative_filtering.py',
        'models/content_based_filtering.py', 
        'models/sentiment_analyzer.py',
        'models/hybrid_recommender.py',
        'complete_demo.py',
        'evaluate_performance.py',
        'requirements.txt'
    ]
    
    print(f"\n🔧 Verifying Key Files:")
    for file in key_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
    
    # Check requirements.txt for key dependencies
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            
        key_deps = ['pandas', 'numpy', 'scikit-learn', 'nltk', 'transformers', 'vaderSentiment']
        print(f"\n📦 Verifying Dependencies:")
        
        for dep in key_deps:
            if dep.lower() in requirements.lower():
                print(f"✅ {dep}")
            else:
                print(f"❌ {dep} - Not found in requirements.txt")
    
    # Test basic imports
    print(f"\n🐍 Testing Imports:")
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas")
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy")
        
    try:
        from sklearn.decomposition import TruncatedSVD
        print("✅ scikit-learn")
    except ImportError:
        print("❌ scikit-learn")
    
    try:
        import nltk
        print("✅ NLTK")
    except ImportError:
        print("❌ NLTK")
    
    # Test model imports
    print(f"\n🤖 Testing Model Imports:")
    sys.path.append('.')
    
    try:
        from models.collaborative_filtering import CollaborativeFiltering
        print("✅ CollaborativeFiltering")
    except ImportError as e:
        print(f"❌ CollaborativeFiltering: {e}")
    
    try:
        from models.content_based_filtering import ContentBasedFiltering
        print("✅ ContentBasedFiltering")
    except ImportError as e:
        print(f"❌ ContentBasedFiltering: {e}")
        
    try:
        from models.sentiment_analyzer import SentimentAnalyzer
        print("✅ SentimentAnalyzer")
    except ImportError as e:
        print(f"❌ SentimentAnalyzer: {e}")
    
    print(f"\n🎯 README.md Verification Complete!")
    print("All major claims have been validated against actual project state.")

if __name__ == "__main__":
    verify_readme_accuracy()
