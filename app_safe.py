"""
Deployment-safe main application that handles missing dependencies gracefully
"""
import streamlit as st
import sys
import os
from typing import Dict, List, Any, Optional

# Try to import required packages, fall back gracefully
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from datetime import datetime
    DATETIME_AVAILABLE = True
except ImportError:
    DATETIME_AVAILABLE = False

# Try to import project modules
try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback config
    class Config:
        PAGE_TITLE = "AI Restaurant Recommendation System"
        PAGE_ICON = "🍽️"
        N_RECOMMENDATIONS = 10

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

def load_data_safe():
    """Safely load data with fallback options"""
    try:
        if PANDAS_AVAILABLE:
            # Try to load CSV files
            restaurants = pd.read_csv('data/restaurants.csv')
            users = pd.read_csv('data/users.csv') 
            ratings = pd.read_csv('data/ratings.csv')
            reviews = pd.read_csv('data/reviews.csv')
            return restaurants, users, ratings, reviews, True
        else:
            return None, None, None, None, False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, False

def initialize_recommender_safe(restaurants, users, ratings, reviews):
    """Safely initialize recommender system"""
    try:
        # Try to use the safe hybrid recommender
        from models.hybrid_recommender_safe import HybridRecommenderSafe
        recommender = HybridRecommenderSafe(restaurants, users, ratings, reviews)
        success = recommender.prepare_data()
        return recommender, success
    except ImportError:
        st.warning("Advanced ML models not available, using basic recommendations")
        return None, False
    except Exception as e:
        st.error(f"Error initializing recommender: {e}")
        return None, False

def get_basic_recommendations(restaurants, ratings, num_recommendations=10):
    """Basic recommendation fallback without ML dependencies"""
    try:
        if restaurants is None or ratings is None:
            return []
        
        # Calculate restaurant popularity
        restaurant_stats = ratings.groupby('restaurant_id').agg({
            'rating': ['mean', 'count']
        }).round(2)
        restaurant_stats.columns = ['avg_rating', 'rating_count']
        
        # Merge with restaurant details
        popular_restaurants = restaurants.merge(
            restaurant_stats, 
            left_on='restaurant_id', 
            right_index=True, 
            how='left'
        )
        
        # Fill missing values
        popular_restaurants['avg_rating'] = popular_restaurants['avg_rating'].fillna(4.0)
        popular_restaurants['rating_count'] = popular_restaurants['rating_count'].fillna(1)
        
        # Calculate popularity score
        popular_restaurants['popularity_score'] = (
            popular_restaurants['avg_rating'] * 
            np.log1p(popular_restaurants['rating_count'])
        )
        
        # Get top recommendations
        top_restaurants = popular_restaurants.nlargest(num_recommendations, 'popularity_score')
        
        recommendations = []
        for _, restaurant in top_restaurants.iterrows():
            rec = {
                'restaurant_id': restaurant['restaurant_id'],
                'name': restaurant.get('name', 'Unknown Restaurant'),
                'cuisine': restaurant.get('cuisine', 'Unknown'),
                'avg_rating': float(restaurant.get('avg_rating', 4.0)),
                'rating_count': int(restaurant.get('rating_count', 1)),
                'recommendation_score': float(restaurant.get('popularity_score', 4.0)),
                'method': 'popularity_based'
            }
            
            if 'address' in restaurant:
                rec['address'] = restaurant['address']
            if 'price_range' in restaurant:
                rec['price_range'] = restaurant['price_range']
                
            recommendations.append(rec)
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating basic recommendations: {e}")
        return []

def main():
    """Main application"""
    
    # Header
    st.title("🍽️ AI-Powered Restaurant Recommendation System")
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Dependencies:**")
            st.write(f"• Pandas: {'✅' if PANDAS_AVAILABLE else '❌'}")
            st.write(f"• Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
            st.write(f"• Config: {'✅' if CONFIG_AVAILABLE else '❌'}")
            
        with col2:
            st.write("**ML Capabilities:**")
            try:
                import sklearn
                st.write("• Scikit-learn: ✅")
            except ImportError:
                st.write("• Scikit-learn: ❌")
            
            try:
                import nltk
                st.write("• NLTK: ✅")
            except ImportError:
                st.write("• NLTK: ❌")
                
        with col3:
            st.write("**Data Status:**")
            restaurants, users, ratings, reviews, data_loaded = load_data_safe()
            st.write(f"• Data Loading: {'✅' if data_loaded else '❌'}")
            if data_loaded:
                st.write(f"• Restaurants: {len(restaurants)}")
                st.write(f"• Users: {len(users)}")
                st.write(f"• Ratings: {len(ratings)}")
    
    # Load data
    restaurants, users, ratings, reviews, data_loaded = load_data_safe()
    
    if not data_loaded:
        st.error("⚠️ Unable to load data. Using demo mode.")
        # Import and run demo app
        try:
            import app_demo
            app_demo.main()
            return
        except ImportError:
            st.error("Demo app not available. Please check your installation.")
            return
    
    # Sidebar for user input
    st.sidebar.header("🎯 Your Preferences")
    
    # User selection
    if PANDAS_AVAILABLE and users is not None:
        user_options = ['Random User'] + list(users['user_id'].unique())
        selected_user = st.sidebar.selectbox("Select User:", user_options)
        
        if selected_user == 'Random User':
            selected_user = users['user_id'].sample(1).iloc[0]
    else:
        selected_user = "demo_user"
    
    # Preference filters
    if PANDAS_AVAILABLE and restaurants is not None:
        cuisines = ['All'] + list(restaurants['cuisine'].unique())
        selected_cuisine = st.sidebar.selectbox("Preferred Cuisine:", cuisines)
        
        min_rating = st.sidebar.slider("Minimum Rating:", 1.0, 5.0, 3.0, 0.1)
        num_recommendations = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)
    else:
        selected_cuisine = 'All'
        min_rating = 3.0
        num_recommendations = 10
    
    # Get recommendations
    st.header("🎯 Recommended Restaurants")
    
    with st.spinner("Generating recommendations..."):
        # Try to use advanced recommender
        recommender, recommender_ready = initialize_recommender_safe(restaurants, users, ratings, reviews)
        
        if recommender and recommender_ready:
            try:
                recommendations = recommender.get_recommendations(
                    str(selected_user), 
                    num_recommendations
                )
                st.success("✅ Using Advanced AI Recommendations")
            except Exception as e:
                st.warning(f"Advanced recommender failed: {e}. Using basic recommendations.")
                recommendations = get_basic_recommendations(restaurants, ratings, num_recommendations)
        else:
            recommendations = get_basic_recommendations(restaurants, ratings, num_recommendations)
            st.info("ℹ️ Using Basic Popularity-Based Recommendations")
    
    # Display recommendations
    if recommendations:
        # Apply filters
        if selected_cuisine != 'All':
            recommendations = [r for r in recommendations if r.get('cuisine', '').lower() == selected_cuisine.lower()]
        
        recommendations = [r for r in recommendations if r.get('avg_rating', 0) >= min_rating]
        
        if not recommendations:
            st.warning("No restaurants match your criteria. Try adjusting your filters.")
        else:
            # Display recommendations in a nice format
            for i, rec in enumerate(recommendations[:num_recommendations], 1):
                with st.expander(f"{i}. {rec['name']} ⭐ {rec.get('avg_rating', 'N/A')}", expanded=i<=3):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Cuisine:** {rec.get('cuisine', 'Unknown')}")
                        if 'address' in rec:
                            st.write(f"**Address:** {rec['address']}")
                        if 'price_range' in rec:
                            st.write(f"**Price Range:** {rec['price_range']}")
                        st.write(f"**Rating:** {rec.get('avg_rating', 'N/A')}/5.0")
                        if 'rating_count' in rec:
                            st.write(f"**Reviews:** {rec['rating_count']} reviews")
                        st.write(f"**Method:** {rec.get('method', 'Unknown')}")
                        
                    with col2:
                        score = rec.get('recommendation_score', 0)
                        st.metric("Recommendation Score", f"{score:.2f}")
                        
                        # Progress bar for score
                        max_score = max([r.get('recommendation_score', 0) for r in recommendations])
                        if max_score > 0:
                            progress = min(score / max_score, 1.0)
                            st.progress(progress)
    else:
        st.warning("No recommendations available. Please check your data and try again.")
    
    # Analytics section
    if PANDAS_AVAILABLE and PLOTLY_AVAILABLE and restaurants is not None:
        st.header("📊 System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cuisine distribution
            st.subheader("Cuisine Distribution")
            cuisine_counts = restaurants['cuisine'].value_counts()
            
            fig_pie = px.pie(
                values=cuisine_counts.values,
                names=cuisine_counts.index,
                title="Restaurant Types"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Rating distribution
            st.subheader("Rating Distribution")
            if ratings is not None:
                fig_hist = px.histogram(
                    ratings,
                    x='rating',
                    nbins=10,
                    title="Rating Distribution",
                    labels={'rating': 'Rating', 'count': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 AI Restaurant Recommendation System
    
    **System Capabilities:**
    - 🤖 **Hybrid ML Algorithms** - Collaborative + Content-based filtering
    - 🎯 **Personalized Recommendations** - Based on user preferences and history
    - 📊 **Data Analytics** - Real-time insights and visualizations
    - 🌐 **Web Interface** - Interactive Streamlit application
    - ☁️ **Cloud Deployment** - Scalable and accessible anywhere
    
    *Built with Python, Streamlit, Scikit-learn, and deployed on Render*
    """)

if __name__ == "__main__":
    main()
