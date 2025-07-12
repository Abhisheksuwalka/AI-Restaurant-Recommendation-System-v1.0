#!/usr/bin/env python3
"""
Test script for LLM-enhanced recommendation system
"""

import sys
import os
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test basic package imports"""
    print("Testing basic imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_llm_imports():
    """Test LLM-related imports"""
    print("Testing LLM imports...")
    try:
        import smolagents
        from dotenv import load_dotenv
        print("✅ LLM imports successful")
        return True
    except ImportError as e:
        print(f"❌ LLM import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from config import Config
        print(f"✅ Config loaded successfully")
        print(f"   LLM Enhancement: {Config.USE_LLM_ENHANCEMENT}")
        print(f"   GitHub Token: {'Configured' if Config.GITHUB_TOKEN else 'Not configured'}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_data_loading():
    """Test data loading and preprocessing"""
    print("Testing data loading...")
    try:
        from data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_restaurants()
        data = preprocessor.get_processed_data()
        
        print(f"✅ Data loaded successfully")
        print(f"   Users: {len(data['users'])}")
        print(f"   Restaurants: {len(data['restaurants'])}")
        print(f"   Ratings: {len(data['ratings'])}")
        print(f"   Reviews: {len(data['reviews'])}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_basic_recommender():
    """Test basic recommendation system without LLM"""
    print("Testing basic recommender...")
    try:
        from data.preprocessor import DataPreprocessor
        from models.hybrid_recommender import HybridRecommender
        
        # Load data
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_restaurants()
        data = preprocessor.get_processed_data()
        
        # Train recommender without LLM
        recommender = HybridRecommender()
        recommender.fit(data)
        
        # Get recommendations for first user
        user_id = data['users']['user_id'].iloc[0]
        recommendations = recommender.get_hybrid_recommendations(user_id, 5)
        
        print(f"✅ Basic recommender working")
        print(f"   Generated {len(recommendations)} recommendations for user {user_id}")
        return True
    except Exception as e:
        print(f"❌ Basic recommender failed: {e}")
        traceback.print_exc()
        return False

def test_llm_recommender():
    """Test LLM-enhanced recommender"""
    print("Testing LLM-enhanced recommender...")
    try:
        from config import Config
        
        if not Config.USE_LLM_ENHANCEMENT:
            print("⚠️  LLM enhancement disabled in config")
            return True
            
        if not Config.GITHUB_TOKEN:
            print("⚠️  GitHub token not configured - skipping LLM test")
            return True
        
        from models.llm_recommender import LLMEnhancedRecommender
        
        # Test LLM enhancer initialization
        llm_enhancer = LLMEnhancedRecommender()
        
        # Test cuisine recommendations
        user_profile = {
            'preferred_cuisine': 'Italian',
            'location': 'Mumbai',
            'budget_preference': 'Medium'
        }
        available_cuisines = ['Italian', 'Chinese', 'Indian', 'Mexican', 'Thai']
        
        cuisine_rec = llm_enhancer.generate_cuisine_recommendations(user_profile, available_cuisines)
        
        print(f"✅ LLM recommender working")
        print(f"   Cuisine recommendation generated: {len(cuisine_rec)} characters")
        return True
    except Exception as e:
        print(f"❌ LLM recommender failed: {e}")
        traceback.print_exc()
        return False

def test_full_system():
    """Test the complete system integration"""
    print("Testing full system integration...")
    try:
        from data.preprocessor import DataPreprocessor
        from models.hybrid_recommender import HybridRecommender
        
        # Load data
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_restaurants()
        data = preprocessor.get_processed_data()
        
        # Train full recommender
        recommender = HybridRecommender()
        recommender.fit(data)
        
        # Get enhanced recommendations
        user_id = data['users']['user_id'].iloc[0]
        recommendations = recommender.get_hybrid_recommendations(user_id, 3)
        
        print(f"✅ Full system working")
        print(f"   Generated {len(recommendations)} enhanced recommendations")
        
        # Check if LLM explanations are present
        if recommendations and 'llm_explanation' in recommendations[0]:
            print(f"   ✅ LLM explanations included")
        else:
            print(f"   ⚠️  LLM explanations not present (may be disabled)")
        
        return True
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 AI Recommendation System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("LLM Imports", test_llm_imports),
        ("Configuration", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Basic Recommender", test_basic_recommender),
        ("LLM Recommender", test_llm_recommender),
        ("Full System", test_full_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
