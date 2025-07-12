#!/bin/bash

# Download NLTK data
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded successfully')
"

# Start the Streamlit app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
