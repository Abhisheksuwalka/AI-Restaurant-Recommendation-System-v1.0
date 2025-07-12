import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data.preprocessor import DataPreprocessor
from models.hybrid_recommender import HybridRecommender
from config import Config

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data with caching"""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.preprocess_restaurants()
    return preprocessor.get_processed_data()

@st.cache_resource
def train_recommender(_data):
    """Train the hybrid recommender with caching"""
    recommender = HybridRecommender()
    recommender.fit(_data)
    return recommender

def main():
    st.title("🍽️ AI-Powered Restaurant Recommendation System")
    st.markdown("### Enhanced with LLM Intelligence & Emotional AI 🤖💭")
    st.markdown("---")
    
    # Load data and train models
    with st.spinner("Loading data and training models..."):
        data = load_and_preprocess_data()
        recommender = train_recommender(data)
    
    # Check enhancement statuses
    llm_status = "✅ Enabled" if recommender.llm_enhancer else "❌ Disabled"
    emotional_status = "✅ Enabled" if recommender.emotional_engine else "❌ Disabled"
    
    st.sidebar.info(f"LLM Enhancement: {llm_status}")
    st.sidebar.info(f"Emotional AI: {emotional_status}")
    
    # Sidebar for user input
    st.sidebar.header("User Preferences")
    
    # User selection
    user_ids = data['users']['user_id'].tolist()
    selected_user = st.sidebar.selectbox("Select User ID", user_ids)
    
    # Emotional input section
    if recommender.emotional_engine:
        st.sidebar.subheader("💭 How are you feeling?")
        user_text_input = st.sidebar.text_area(
            "Tell us about your mood or what you're looking for:",
            placeholder="e.g., I'm feeling stressed and need comfort food...",
            height=100
        )
        
        # Location input for contextual analysis
        user_location = st.sidebar.text_input(
            "Your location (optional):",
            placeholder="e.g., New York, NY",
            help="Used for weather-based mood analysis"
        )
        
        # Recommendation mode selection
        rec_mode = st.sidebar.radio(
            "Recommendation Mode:",
            ["🧠 Emotional AI", "🔄 Hybrid (All Features)", "📊 Traditional"],
            index=1
        )
    else:
        user_text_input = ""
        user_location = None
        rec_mode = "📊 Traditional"
    
    # Recommendation parameters
    st.sidebar.subheader("Recommendation Settings")
    n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # Weight sliders (adjust based on emotional intelligence availability)
    st.sidebar.subheader("Algorithm Weights")
    if recommender.emotional_engine and rec_mode == "🔄 Hybrid (All Features)":
        collab_weight = st.sidebar.slider("Collaborative Filtering", 0.0, 1.0, 0.3, 0.1)
        content_weight = st.sidebar.slider("Content-Based", 0.0, 1.0, 0.2, 0.1)
        sentiment_weight = st.sidebar.slider("Sentiment Analysis", 0.0, 1.0, 0.2, 0.1)
        emotional_weight = st.sidebar.slider("Emotional Intelligence", 0.0, 1.0, 0.3, 0.1)
        
        # Normalize weights
        total_weight = collab_weight + content_weight + sentiment_weight + emotional_weight
        if total_weight > 0:
            collab_weight /= total_weight
            content_weight /= total_weight
            sentiment_weight /= total_weight
            emotional_weight /= total_weight
    else:
        collab_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.4, 0.1)
        content_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.3, 0.1)
        sentiment_weight = st.sidebar.slider("Sentiment Analysis Weight", 0.0, 1.0, 0.3, 0.1)
        emotional_weight = 0.0
        
        # Normalize weights
        total_weight = collab_weight + content_weight + sentiment_weight
        if total_weight > 0:
            collab_weight /= total_weight
            content_weight /= total_weight
            sentiment_weight /= total_weight
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Personalized Recommendations")
        
        # Emotional intelligence features
        if recommender.emotional_engine:
            # Mood-based cuisine suggestions
            with st.expander("🎭 Mood-Based Cuisine Suggestions", expanded=False):
                if st.button("Get Mood-Based Cuisines"):
                    with st.spinner("Analyzing your mood..."):
                        mood_suggestions = recommender.get_mood_based_cuisine_suggestions(
                            selected_user, user_text_input, user_location
                        )
                        if isinstance(mood_suggestions, dict):
                            st.success(f"**Detected Emotion:** {mood_suggestions['detected_emotion'].title()} "
                                     f"(Intensity: {mood_suggestions['emotion_intensity']:.1%})")
                            st.markdown(mood_suggestions['explanation'])
                            
                            st.subheader("Recommended Cuisines:")
                            for cuisine_info in mood_suggestions['recommended_cuisines']:
                                compatibility = cuisine_info['emotional_compatibility']
                                st.write(f"🍽️ **{cuisine_info['cuisine'].title()}** "
                                       f"(Match: {compatibility:.1%}) - {cuisine_info['reason']}")
                        else:
                            st.info(mood_suggestions)
        
        # Add LLM-powered cuisine recommendations
        if recommender.llm_enhancer:
            with st.expander("🤖 AI Cuisine Suggestions", expanded=False):
                if st.button("Get AI Cuisine Recommendations"):
                    with st.spinner("Generating AI recommendations..."):
                        cuisine_suggestions = recommender.get_cuisine_recommendations(selected_user)
                        st.markdown(cuisine_suggestions)
        
        # Recommendation button based on mode
        if rec_mode == "🧠 Emotional AI":
            button_text = "Get Emotional AI Recommendations"
            if not user_text_input:
                st.info("💡 Tip: Share how you're feeling in the sidebar for better emotional recommendations!")
        elif rec_mode == "🔄 Hybrid (All Features)":
            button_text = "Get Hybrid Recommendations"
        else:
            button_text = "Get Traditional Recommendations"
        
        if st.button(button_text, type="primary"):
            with st.spinner("Generating recommendations..."):
                if rec_mode == "🧠 Emotional AI":
                    recommendations = recommender.get_emotional_recommendations(
                        selected_user, user_text_input, user_location, n_recommendations
                    )
                elif rec_mode == "🔄 Hybrid (All Features)":
                    recommendations = recommender.get_hybrid_recommendations(
                        selected_user, n_recommendations, collab_weight, content_weight, 
                        sentiment_weight, emotional_weight, user_text_input, user_location
                    )
                else:
                    recommendations = recommender.get_hybrid_recommendations(
                        selected_user, n_recommendations, collab_weight, content_weight, sentiment_weight
                    )
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['name']} ⭐ {rec['rating']}", expanded=i<=3):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.write(f"**Cuisine:** {rec['cuisine']}")
                            st.write(f"**Location:** {rec['location']}")
                            st.write(f"**Price Range:** {rec['price_range']}")
                        
                        with col_b:
                            st.write(f"**Final Score:** {rec['final_score']:.3f}")
                            st.write(f"**Sentiment Score:** {rec['sentiment_score']:.3f}")
                            if 'emotional_score' in rec:
                                st.write(f"**Emotional Score:** {rec['emotional_score']:.3f}")
                        
                        with col_c:
                            sentiment_info = rec['sentiment_info']
                            if sentiment_info.get('total_reviews', 0) > 0:
                                st.write(f"**Reviews:** {sentiment_info['total_reviews']}")
                                st.write(f"**Positive:** {sentiment_info['positive_ratio']:.1%}")
                                st.write(f"**Negative:** {sentiment_info['negative_ratio']:.1%}")
                        
                        # Display emotional explanation if available
                        if 'emotional_explanation' in rec and rec['emotional_explanation']:
                            st.markdown("**💭 Emotional Match:**")
                            st.info(rec['emotional_explanation'])
                        
                        # Display LLM explanation if available
                        if 'llm_explanation' in rec and rec['llm_explanation']:
                            st.markdown("**🤖 AI Explanation:**")
                            st.success(rec['llm_explanation'])
                        
                        # Show detected emotion and intensity for emotional recommendations
                        if 'detected_emotion' in rec and 'emotion_intensity' in rec:
                            st.markdown(f"**🎭 Detected Mood:** {rec['detected_emotion'].title()} "
                                      f"(Intensity: {rec['emotion_intensity']:.1%})")
                        
                        # Add review summary button
                        if recommender.llm_enhancer:
                            if st.button(f"Get AI Review Summary", key=f"summary_{rec['restaurant_id']}"):
                                with st.spinner("Analyzing reviews..."):
                                    review_summary = recommender.get_restaurant_review_summary(rec['restaurant_id'])
                                    st.markdown("**📝 Review Summary:**")
                                    st.success(review_summary)
            else:
                st.warning("No recommendations found for this user.")
    
    with col2:
        st.header("📊 User Profile")
        
        # Display user information
        user_info = data['users'][data['users']['user_id'] == selected_user].iloc[0]
        st.write(f"**Age:** {user_info['age']}")
        st.write(f"**Location:** {user_info['location']}")
        st.write(f"**Preferred Cuisine:** {user_info['preferred_cuisine']}")
        st.write(f"**Price Preference:** {user_info['price_preference']}")
        
        # Emotional insights section
        if recommender.emotional_engine:
            st.subheader("💭 Emotional Insights")
            emotional_insights = recommender.get_user_emotional_insights(selected_user)
            
            if "status" not in emotional_insights:
                col_emo1, col_emo2 = st.columns(2)
                
                with col_emo1:
                    st.metric("Emotions Tracked", emotional_insights.get('emotional_history_count', 0))
                    if emotional_insights.get('most_frequent_emotion'):
                        st.metric("Most Frequent Mood", emotional_insights['most_frequent_emotion'].title())
                
                with col_emo2:
                    if emotional_insights.get('last_emotion_detected'):
                        st.metric("Last Detected Mood", emotional_insights['last_emotion_detected'].title())
                        
                        # Display recent emotions
                        if emotional_insights.get('recent_emotions'):
                            st.write("**Recent Moods:**")
                            recent_emotions = emotional_insights['recent_emotions'][:3]
                            mood_emojis = {
                                'happy': '😊', 'sad': '😢', 'stressed': '😰', 'excited': '🤩',
                                'anxious': '😟', 'romantic': '😍', 'energetic': '⚡', 'calm': '😌',
                                'adventurous': '🗺️', 'nostalgic': '💭', 'neutral': '😐'
                            }
                            for emotion in recent_emotions:
                                emoji = mood_emojis.get(emotion, '🙂')
                                st.write(f"{emoji} {emotion.title()}")
                
                # Emotional patterns chart
                if emotional_insights.get('emotional_patterns'):
                    patterns = emotional_insights['emotional_patterns']
                    emotions = list(patterns.keys())
                    frequencies = list(patterns.values())
                    
                    fig_emotions = px.bar(
                        x=emotions, y=frequencies,
                        title="Emotional Patterns",
                        labels={'x': 'Emotions', 'y': 'Frequency'}
                    )
                    fig_emotions.update_layout(height=300)
                    st.plotly_chart(fig_emotions, use_container_width=True)
            else:
                st.info(emotional_insights['status'])
        
        # User's rating history
        st.subheader("📈 Rating History")
        user_ratings = data['ratings'][data['ratings']['user_id'] == selected_user]
        
        if len(user_ratings) > 0:
            avg_rating = user_ratings['rating'].mean()
            total_ratings = len(user_ratings)
            
            st.metric("Average Rating", f"{avg_rating:.1f}", f"{total_ratings} restaurants")
            
            # Rating distribution
            rating_dist = user_ratings['rating'].value_counts().sort_index()
            fig = px.bar(x=rating_dist.index, y=rating_dist.values, 
                        title="Rating Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating history available for this user.")
    
    # Analytics section
    st.markdown("---")
    st.header("📊 System Analytics")
    
    if recommender.emotional_engine:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Restaurant Analytics", "Sentiment Analysis", "Algorithm Performance", 
            "LLM Insights", "🧠 Emotional Intelligence"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Restaurant Analytics", "Sentiment Analysis", "Algorithm Performance", "LLM Insights"
        ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cuisine distribution
            cuisine_dist = data['restaurants']['cuisine'].value_counts()
            fig = px.pie(values=cuisine_dist.values, names=cuisine_dist.index, 
                        title="Restaurant Distribution by Cuisine")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            rating_dist = data['restaurants']['rating'].hist()
            fig = px.histogram(data['restaurants'], x='rating', 
                             title="Restaurant Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Sentiment analysis results
        if hasattr(recommender, 'sentiment_results'):
            sentiment_dist = recommender.sentiment_results['vader_sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=sentiment_dist.values, names=sentiment_dist.index,
                           title="Overall Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sentiment vs Rating correlation
                fig = px.scatter(recommender.sentiment_results, 
                               x='rating', y='vader_compound',
                               title="Rating vs Sentiment Score")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Algorithm Weights")
        weights_df = pd.DataFrame({
            'Algorithm': ['Collaborative Filtering', 'Content-Based', 'Sentiment Analysis'],
            'Weight': [collab_weight, content_weight, sentiment_weight]
        })
        
        fig = px.bar(weights_df, x='Algorithm', y='Weight', 
                    title="Current Algorithm Weights")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", len(data['users']))
        with col2:
            st.metric("Total Restaurants", len(data['restaurants']))
        with col3:
            st.metric("Total Ratings", len(data['ratings']))
    
    with tab4:
        st.subheader("🤖 LLM Enhancement Status")
        
        if recommender.llm_enhancer:
            st.success("LLM Enhancement is Active")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Model:** {Config.LLM_MODEL}")
                st.info(f"**Max Tokens:** {Config.MAX_TOKENS}")
                st.info(f"**Temperature:** {Config.TEMPERATURE}")
            
            with col2:
                st.info(f"**Explanations:** {'Enabled' if Config.LLM_EXPLANATION_ENABLED else 'Disabled'}")
                st.info(f"**GitHub API:** {'Connected' if Config.GITHUB_TOKEN else 'Not configured'}")
            
            # LLM feature showcase
            st.subheader("🚀 AI-Powered Features")
            feature_data = {
                'Feature': ['Personalized Explanations', 'Cuisine Recommendations', 'Review Summaries', 'Context-aware Suggestions'],
                'Status': ['✅ Active', '✅ Active', '✅ Active', '✅ Active'],
                'Description': [
                    'AI explains why each restaurant is recommended',
                    'Smart cuisine suggestions based on user profile',
                    'Automated review analysis and summarization',
                    'Time and context-aware dining suggestions'
                ]
            }
            
            features_df = pd.DataFrame(feature_data)
            st.dataframe(features_df, use_container_width=True)
            
        else:
            st.warning("LLM Enhancement is not available")
            st.info("To enable LLM features, please configure your GitHub token in the .env file")
            
            # Show what's missing
            issues = []
            if not Config.GITHUB_TOKEN:
                issues.append("❌ GitHub token not configured")
            
            if issues:
                st.error("Configuration Issues:")
                for issue in issues:
                    st.write(issue)
    
    # Emotional Intelligence tab (only show if emotional engine is available)
    if recommender.emotional_engine:
        with tab5:
            st.subheader("🧠 Emotional Intelligence Dashboard")
            
            st.success("Emotional AI is Active")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Emotion Detection:** Transformer + VADER")
                st.info("**Context Analysis:** Weather + Time + Calendar")
                st.info("**Cache TTL:** 1 hour")
                
            with col2:
                st.info(f"**Emotional Weight:** {Config.EMOTIONAL_WEIGHT:.1%}")
                st.info(f"**Supported Emotions:** {len(Config.EMOTIONAL_STATES)}")
                st.info(f"**Weather API:** {'Connected' if Config.WEATHER_API_KEY else 'Not configured'}")
            
            # Show supported emotions
            st.subheader("🎭 Supported Emotional States")
            emotions_grid = st.columns(5)
            emotion_emojis = {
                'happy': '😊', 'sad': '😢', 'stressed': '😰', 'excited': '🤩',
                'anxious': '😟', 'romantic': '😍', 'energetic': '⚡', 'calm': '😌',
                'adventurous': '🗺️', 'nostalgic': '💭'
            }
            
            for i, emotion in enumerate(Config.EMOTIONAL_STATES):
                with emotions_grid[i % 5]:
                    emoji = emotion_emojis.get(emotion, '🙂')
                    st.write(f"{emoji} {emotion.title()}")
            
            # Emotional features showcase
            st.subheader("🚀 Emotional AI Features")
            emotional_features = {
                'Feature': [
                    'Real-time Emotion Detection',
                    'Context-Aware Analysis', 
                    'Mood-Based Recommendations',
                    'Emotional History Tracking',
                    'Cuisine-Emotion Mapping',
                    'Weather Influence Analysis'
                ],
                'Status': ['✅ Active'] * 6,
                'Description': [
                    'Detects emotions from text input using advanced NLP',
                    'Considers time, weather, and calendar for context',
                    'Recommends restaurants based on current emotional state',
                    'Tracks user emotional patterns over time',
                    'Maps cuisines to emotional states for better matching',
                    'Adjusts recommendations based on weather conditions'
                ]
            }
            
            emotional_features_df = pd.DataFrame(emotional_features)
            st.dataframe(emotional_features_df, use_container_width=True)
            
            # Demo section
            st.subheader("🎪 Try Emotional Detection")
            demo_text = st.text_input(
                "Enter some text to see emotion detection in action:",
                placeholder="e.g., I'm feeling stressed about work..."
            )
            
            if demo_text and st.button("Analyze Emotion"):
                with st.spinner("Analyzing emotion..."):
                    try:
                        emotional_state = recommender.emotional_engine.state_manager.detect_current_emotion(
                            "demo_user", demo_text
                        )
                        
                        col_demo1, col_demo2, col_demo3 = st.columns(3)
                        
                        with col_demo1:
                            st.metric("Primary Emotion", emotional_state.primary_emotion.title())
                        
                        with col_demo2:
                            st.metric("Intensity", f"{emotional_state.intensity:.1%}")
                        
                        with col_demo3:
                            st.metric("Confidence", f"{emotional_state.confidence:.1%}")
                        
                        if emotional_state.secondary_emotion:
                            st.info(f"Secondary emotion detected: {emotional_state.secondary_emotion.title()}")
                        
                    except Exception as e:
                        st.error(f"Error analyzing emotion: {str(e)}")

if __name__ == "__main__":
    main()
