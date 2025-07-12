#!/usr/bin/env python3
"""
NLTK Data Downloader for Production Deployment
Downloads essential NLTK data packages required for sentiment analysis
"""

import sys
import ssl
import os

def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        import nltk
        
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set NLTK data path
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required packages
        packages = [
            'vader_lexicon',
            'punkt', 
            'stopwords',
            'averaged_perceptron_tagger',
            'wordnet'
        ]
        
        success_count = 0
        for package in packages:
            try:
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)
                print(f"✅ Downloaded: {package}")
                success_count += 1
            except Exception as e:
                print(f"⚠️  Failed to download {package}: {e}")
        
        if success_count > 0:
            print(f"✅ NLTK: {success_count}/{len(packages)} packages downloaded successfully")
            return True
        else:
            print("❌ NLTK: No packages downloaded successfully")
            return False
            
    except ImportError:
        print("⚠️  NLTK not available, skipping download")
        return False
    except Exception as e:
        print(f"❌ NLTK download error: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)
