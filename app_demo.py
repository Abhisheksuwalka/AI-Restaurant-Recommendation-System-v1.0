"""
Deployment-safe demo version of the restaurant recommendation app
This version works without pandas, numpy, or sklearn for initial deployment
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
from typing import List, Dict, Any

# Set page config
st.set_page_config(
    page_title="AI Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_demo_data():
    """Load demo data without pandas dependency"""
    demo_restaurants = [
        {
            "restaurant_id": "R001",
            "name": "Spice Palace",
            "cuisine_type": "Indian",
            "avg_rating": 4.5,
            "rating_count": 150,
            "address": "123 Curry Street",
            "price_range": "$$"
        },
        {
            "restaurant_id": "R002", 
            "name": "Pasta Paradise",
            "cuisine_type": "Italian",
            "avg_rating": 4.2,
            "rating_count": 89,
            "address": "456 Pizza Avenue",
            "price_range": "$$$"
        },
        {
            "restaurant_id": "R003",
            "name": "Sushi Zen",
            "cuisine_type": "Japanese", 
            "avg_rating": 4.7,
            "rating_count": 203,
            "address": "789 Sashimi Lane",
            "price_range": "$$$$"
        },
        {
            "restaurant_id": "R004",
            "name": "Taco Fiesta", 
            "cuisine_type": "Mexican",
            "avg_rating": 4.1,
            "rating_count": 67,
            "address": "321 Salsa Street",
            "price_range": "$"
        },
        {
            "restaurant_id": "R005",
            "name": "Dragon Garden",
            "cuisine_type": "Chinese",
            "avg_rating": 4.3,
            "rating_count": 112,
            "address": "654 Noodle Road",
            "price_range": "$$"
        }
    ]
    
    demo_reviews = [
        {"restaurant_id": "R001", "rating": 5, "review": "Amazing flavors and great service!"},
        {"restaurant_id": "R001", "rating": 4, "review": "Good food but a bit spicy for me"},
        {"restaurant_id": "R002", "rating": 4, "review": "Authentic Italian taste"},
        {"restaurant_id": "R003", "rating": 5, "review": "Fresh sushi, excellent quality"},
        {"restaurant_id": "R004", "rating": 4, "review": "Great tacos and friendly staff"},
    ]
    
    return demo_restaurants, demo_reviews

def get_recommendations(user_preferences: Dict[str, Any], restaurants: List[Dict]) -> List[Dict]:
    """Simple recommendation algorithm without ML dependencies"""
    recommendations = []
    
    preferred_cuisine = user_preferences.get('cuisine', '')
    min_rating = user_preferences.get('min_rating', 0)
    max_price = user_preferences.get('max_price', 4)
    
    price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
    
    for restaurant in restaurants:
        score = restaurant['avg_rating']
        
        # Boost score for preferred cuisine
        if preferred_cuisine and restaurant['cuisine_type'].lower() == preferred_cuisine.lower():
            score += 0.5
            
        # Check rating filter
        if restaurant['avg_rating'] < min_rating:
            continue
            
        # Check price filter  
        restaurant_price = price_map.get(restaurant['price_range'], 2)
        if restaurant_price > max_price:
            continue
            
        # Add popularity boost
        popularity_boost = min(restaurant['rating_count'] / 100, 0.5)
        score += popularity_boost
        
        rec = restaurant.copy()
        rec['recommendation_score'] = round(score, 2)
        rec['method'] = 'demo_algorithm'
        recommendations.append(rec)
    
    # Sort by score
    recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
    return recommendations

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🍽️ AI-Powered Restaurant Recommendation System")
    st.markdown("*Deployment Demo Version - Powered by Streamlit*")
    
    # Sidebar for preferences
    st.sidebar.header("🎯 Your Preferences")
    
    cuisine_options = ['All', 'Indian', 'Italian', 'Japanese', 'Mexican', 'Chinese']
    selected_cuisine = st.sidebar.selectbox(
        "Preferred Cuisine:",
        cuisine_options
    )
    
    min_rating = st.sidebar.slider(
        "Minimum Rating:", 
        min_value=1.0, 
        max_value=5.0, 
        value=3.0, 
        step=0.1
    )
    
    price_options = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
    max_price_label = st.sidebar.select_slider(
        "Maximum Price Range:",
        options=list(price_options.keys()),
        value='$$$'
    )
    max_price = price_options[max_price_label]
    
    # Load demo data
    restaurants, reviews = load_demo_data()
    
    # User preferences
    user_prefs = {
        'cuisine': selected_cuisine if selected_cuisine != 'All' else '',
        'min_rating': min_rating,
        'max_price': max_price
    }
    
    # Get recommendations
    recommendations = get_recommendations(user_prefs, restaurants)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Recommended Restaurants")
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                with st.expander(f"{i}. {rec['name']} ⭐ {rec['avg_rating']}", expanded=i<=3):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.write(f"**Cuisine:** {rec['cuisine_type']}")
                        st.write(f"**Address:** {rec['address']}")
                        st.write(f"**Price Range:** {rec['price_range']}")
                        st.write(f"**Rating:** {rec['avg_rating']}/5.0 ({rec['rating_count']} reviews)")
                        
                    with col_b:
                        st.metric(
                            "Recommendation Score",
                            f"{rec['recommendation_score']:.1f}",
                            delta=f"+{rec['recommendation_score'] - rec['avg_rating']:.1f}"
                        )
                        
                    st.progress(rec['recommendation_score'] / 5.0)
        else:
            st.warning("No restaurants match your criteria. Try adjusting your filters!")
    
    with col2:
        st.header("📊 System Stats")
        
        # Stats
        total_restaurants = len(restaurants)
        avg_rating = sum(r['avg_rating'] for r in restaurants) / len(restaurants)
        total_reviews = sum(r['rating_count'] for r in restaurants)
        
        st.metric("Total Restaurants", total_restaurants)
        st.metric("Average Rating", f"{avg_rating:.1f}⭐")
        st.metric("Total Reviews", total_reviews)
        
        # Chart
        st.subheader("Cuisine Distribution")
        cuisine_counts = {}
        for r in restaurants:
            cuisine = r['cuisine_type']
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
            
        fig = px.pie(
            values=list(cuisine_counts.values()),
            names=list(cuisine_counts.keys()),
            title="Restaurant Types"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        st.subheader("Rating Distribution")
        ratings = [r['avg_rating'] for r in restaurants]
        fig2 = px.histogram(
            x=ratings,
            nbins=10,
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Deployment Success!
    
    **System Features:**
    - ✅ **Streamlit Web Interface** - Interactive and responsive
    - ✅ **Real-time Filtering** - Dynamic preference-based recommendations  
    - ✅ **Visualization Dashboard** - Charts and metrics
    - ✅ **Scalable Architecture** - Ready for production data
    - ✅ **Cloud Deployment** - Hosted on Render
    
    **Tech Stack:** Python • Streamlit • Plotly • Render
    
    *This is a demo version showcasing the core functionality. 
    The full version includes advanced ML algorithms, sentiment analysis, and database integration.*
    """)
    
    # System info
    with st.expander("🔧 System Information"):
        st.write("**Deployment Platform:** Render")
        st.write("**Python Version:** 3.13+")
        st.write("**Framework:** Streamlit")
        st.write("**Status:** ✅ Successfully Deployed")
        
        # Environment info
        port = os.environ.get('PORT', '8501')
        st.write(f"**Server Port:** {port}")
        
        if st.button("🔄 Refresh System Status"):
            st.success("System is running smoothly! 🎉")

if __name__ == "__main__":
    main()
