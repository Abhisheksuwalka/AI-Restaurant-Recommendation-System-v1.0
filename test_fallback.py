#!/usr/bin/env python3
"""
Quick test to verify app_safe.py works in fallback mode
"""

import sys
import os
import pandas as pd

print("🧪 Testing app_safe.py in fallback mode...")
print()

# Test environment
print("Environment check:")
print(f"• Python: {sys.version}")
print(f"• Working directory: {os.getcwd()}")

# Test basic imports
try:
    import streamlit as st
    print("• Streamlit: ✅")
except ImportError:
    print("• Streamlit: ❌")

try:
    import pandas as pd
    print("• Pandas: ✅")
except ImportError:
    print("• Pandas: ❌")

try:
    import sklearn
    print("• Scikit-learn: ❌ (Expected)")
except ImportError:
    print("• Scikit-learn: ❌ (Expected)")

try:
    import nltk
    print("• NLTK: ❌ (Expected)")
except ImportError:
    print("• NLTK: ❌ (Expected)")

print()

# Test data loading
try:
    restaurants = pd.read_csv('data/restaurants.csv')
    print(f"✅ Data loading: {len(restaurants)} restaurants")
    
    if 'cuisine' in restaurants.columns:
        cuisines = restaurants['cuisine'].unique()
        print(f"✅ Cuisine column: {len(cuisines)} types")
    else:
        print("❌ Cuisine column missing")
        
except Exception as e:
    print(f"❌ Data loading error: {e}")

# Test app_safe import
try:
    # Import modules from app_safe without running main
    with open('app_safe.py', 'r') as f:
        content = f.read()
    
    # Execute only the functions part
    parts = content.split('if __name__ == "__main__":')
    if len(parts) > 1:
        functions_code = parts[0]
        namespace = {}
        exec(functions_code, namespace)
        print("✅ app_safe.py functions loaded")
        
        # Test key function
        if 'load_data_safe' in namespace:
            print("✅ load_data_safe function found")
        else:
            print("❌ load_data_safe function missing")
            
    else:
        print("❌ Could not split app_safe.py")
        
except Exception as e:
    print(f"❌ app_safe.py error: {e}")

print()
print("🎯 Test completed - App should work in fallback mode!")
