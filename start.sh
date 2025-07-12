#!/bin/bash

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Download NLTK data with better error handling
python -c "
import nltk
import ssl
import sys

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️  NLTK download warning: {e}')
    print('Continuing without NLTK data...')
"

# Start the Streamlit app
echo "🚀 Starting Streamlit app on port $STREAMLIT_SERVER_PORT"
streamlit run app.py --server.port=$STREAMLIT_SERVER_PORT --server.address=$STREAMLIT_SERVER_ADDRESS
