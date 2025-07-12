# LLM Integration Guide

## Overview

This document explains how to set up and use the LLM-enhanced features in the AI Recommendation System using smolagents and GitHub marketplace models API.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Automated setup (recommended)
python setup.py --all

# Or manual setup
pip install -r requirements.txt
```

### 2. Configure GitHub Token

1. **Get a GitHub Token:**

   - Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
   - Click "Generate new token" (classic or fine-grained)
   - Select appropriate scopes for accessing marketplace models
   - Copy the generated token

2. **Add to Environment:**
   ```bash
   # Edit .env file
   GITHUB_TOKEN=your_github_token_here
   USE_LLM_ENHANCEMENT=true
   LLM_EXPLANATION_ENABLED=true
   ```

### 3. Test the Setup

```bash
# Run the test suite
python test_llm_system.py

# Run a demo
python demo_llm.py

# Start the full application
python run.py
```

## 🧠 LLM Features

### 1. AI-Powered Recommendation Explanations

**What it does:** Generates natural language explanations for why each restaurant is recommended.

**Example:**

```
"This Italian restaurant is perfect for you because it matches your preferred cuisine,
is located in your area (Mumbai), and has excellent reviews praising its authentic
pasta dishes which align with your medium budget preference."
```

**How to use:**

- Enable in config: `LLM_EXPLANATION_ENABLED=true`
- Explanations appear automatically with recommendations
- Each recommendation shows an "🤖 AI Explanation" section

### 2. Smart Cuisine Recommendations

**What it does:** Analyzes user preferences to suggest cuisines they might enjoy.

**Example:**

```
1. Chinese: Great alternative to Italian with similar flavor complexity
2. Mexican: Matches your preference for bold, flavorful dishes
3. Thai: Perfect fusion option given your location and taste profile
```

**How to use:**

- Click "Get AI Cuisine Recommendations" in the sidebar
- Based on user profile, location, and preferences
- Updates dynamically as you change users

### 3. Automated Review Summaries

**What it does:** Analyzes all reviews for a restaurant and creates a balanced summary.

**Example:**

```
Customers consistently praise the authentic flavors and generous portions, with many
highlighting the excellent service. Some mention occasional delays during peak hours,
but overall satisfaction remains high with 90% positive sentiment.
```

**How to use:**

- Click "Get AI Review Summary" button for any restaurant
- Analyzes sentiment, common themes, and patterns
- Provides balanced perspective for decision making

### 4. Context-Aware Suggestions

**What it does:** Considers time, weather, and other contextual factors for recommendations.

**Features:**

- Time-based suggestions (breakfast, lunch, dinner)
- Weather-appropriate cuisine recommendations
- Location and mood-based filtering

## ⚙️ Configuration Options

### Environment Variables

```env
# Core LLM Settings
GITHUB_TOKEN=your_token_here           # Required for LLM features
USE_LLM_ENHANCEMENT=true               # Enable/disable all LLM features
LLM_EXPLANATION_ENABLED=true           # Enable recommendation explanations

# Model Configuration
LLM_MODEL=gpt-4                        # Model to use (gpt-4, gpt-3.5-turbo, etc.)
MAX_TOKENS=1000                        # Maximum tokens per request
TEMPERATURE=0.7                        # Creativity level (0.0 = deterministic, 1.0 = creative)
```

### Supported Models

The system supports various models through GitHub marketplace:

- **GPT-4**: Best quality, slower, higher cost
- **GPT-3.5-turbo**: Good balance of quality and speed
- **Claude-3**: Anthropic's model with strong reasoning
- **Llama models**: Open-source alternatives

### Performance Tuning

```env
# For faster responses
MAX_TOKENS=500
TEMPERATURE=0.3

# For more creative explanations
MAX_TOKENS=1500
TEMPERATURE=0.8
```

## 🛠️ Development

### Adding New LLM Features

1. **Extend LLMEnhancedRecommender:**

   ```python
   def new_llm_feature(self, input_data):
       prompt = f"Your prompt here: {input_data}"
       response = self.agent.run(prompt)
       return response.strip()
   ```

2. **Add to HybridRecommender:**

   ```python
   def get_new_feature(self, user_id):
       if not self.llm_enhancer:
           return "Feature not available"
       return self.llm_enhancer.new_llm_feature(user_data)
   ```

3. **Update UI in app.py:**
   ```python
   if st.button("New Feature"):
       result = recommender.get_new_feature(user_id)
       st.write(result)
   ```

### Error Handling

The system includes robust error handling:

- **Graceful degradation**: If LLM fails, system continues without LLM features
- **Retry logic**: Automatic retries for transient failures
- **Fallback responses**: Default explanations when LLM is unavailable
- **Logging**: Detailed logs for debugging

### Testing

```bash
# Test individual components
python test_llm_system.py

# Test specific features
python -c "from models.llm_recommender import LLMEnhancedRecommender; print('LLM working')"

# Integration test
python demo_llm.py
```

## 🚨 Troubleshooting

### Common Issues

1. **"GitHub token not found"**

   - Check `.env` file exists and has `GITHUB_TOKEN=your_token`
   - Verify token has correct permissions

2. **"smolagents import error"**

   - Install with: `pip install smolagents>=0.4.0`
   - Check Python version compatibility

3. **"Rate limit exceeded"**

   - Add delays between requests
   - Use a different model with higher limits
   - Check your GitHub API quota

4. **"Model not responding"**
   - Verify internet connection
   - Check GitHub marketplace model status
   - Try a different model in config

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

If LLM responses are slow:

1. Reduce `MAX_TOKENS`
2. Use simpler prompts
3. Switch to faster model (e.g., gpt-3.5-turbo)
4. Enable caching for repeated requests

## 📚 Resources

- [smolagents Documentation](https://github.com/huggingface/smolagents)
- [GitHub Marketplace Models](https://github.com/marketplace)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/)

## 🤝 Contributing

To contribute new LLM features:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
