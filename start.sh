#!/bin/bash

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

echo "🚀 Starting AI Restaurant Recommendation System"
echo "🌐 Port: $STREAMLIT_SERVER_PORT"
echo "📍 Address: $STREAMLIT_SERVER_ADDRESS"

# Use the deployment-safe app that handles missing dependencies gracefully
APP_FILE="app_safe.py"
echo "🎯 Using deployment-safe application"

# Download NLTK data if available
python -c "
try:
    import nltk
    import ssl
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
        print('✅ NLTK data downloaded successfully')
    except Exception as e:
        print(f'⚠️  NLTK download warning: {e}')
except ImportError:
    print('⚠️  NLTK not available, skipping download')
"

# Start the Streamlit app
echo "🎬 Launching Streamlit application: $APP_FILE"
streamlit run $APP_FILE --server.port=$STREAMLIT_SERVER_PORT --server.address=$STREAMLIT_SERVER_ADDRESS
