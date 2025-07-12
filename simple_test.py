#!/usr/bin/env python3
"""
Simple test to check core functionality
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports"""
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        from sklearn.preprocessing import LabelEncoder
        print("✓ sklearn imported successfully")
        
        # Test our modules
        from tests.test_factory import TestDataFactory
        print("✓ TestDataFactory imported successfully")
        
        from tests.base_test import BaseTestCase
        print("✓ BaseTestCase imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_factory():
    """Test data factory functionality"""
    try:
        from tests.test_factory import TestDataFactory
        
        factory = TestDataFactory()
        print("✓ TestDataFactory created")
        
        users = factory.create_users(10)
        print(f"✓ Created {len(users)} test users")
        
        restaurants = factory.create_restaurants(20)
        print(f"✓ Created {len(restaurants)} test restaurants")
        
        ratings = factory.create_ratings(users, restaurants, 50)
        print(f"✓ Created {len(ratings)} test ratings")
        
        return True
    except Exception as e:
        print(f"✗ Data factory error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        from config import Config
        print("✓ Config imported successfully")
        print(f"✓ LLM Enhancement: {Config.USE_LLM_ENHANCEMENT}")
        print(f"✓ Emotional AI: {Config.USE_EMOTIONAL_RECOMMENDATIONS}")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def main():
    print("🧪 Running Simple System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Data Factory", test_data_factory), 
        ("Configuration", test_config)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
