#!/usr/bin/env python3
"""
Health check script for Render deployment
"""
import requests
import sys
import os

def health_check():
    """Check if the Streamlit app is running"""
    port = os.environ.get('PORT', '8501')
    url = f"http://localhost:{port}/healthz"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ App is healthy")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1)
