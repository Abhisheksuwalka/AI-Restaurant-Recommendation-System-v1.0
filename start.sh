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
python download_nltk.py

# Start the Streamlit app
echo "🎬 Launching Streamlit application: $APP_FILE"
streamlit run $APP_FILE --server.port=$STREAMLIT_SERVER_PORT --server.address=$STREAMLIT_SERVER_ADDRESS
