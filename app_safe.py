"""
Production-Ready Restaurant Recommendation System
Uses fallback data and built-in algorithms - no external ML dependencies required
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Any, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="🍽️ AI Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide"
)

def load_fallback_data() -> Tuple[bool, Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                  Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load the fallback datasets"""
    try:
        fallback_dir = 'data/fallback'
        
        restaurants = pd.read_csv(os.path.join(fallback_dir, 'restaurants.csv'))
        users = pd.read_csv(os.path.join(fallback_dir, 'users.csv'))
        ratings = pd.read_csv(os.path.join(fallback_dir, 'ratings.csv'))
        reviews = pd.read_csv(os.path.join(fallback_dir, 'reviews.csv'))
        
        return True, restaurants, users, ratings, reviews
        
    except Exception as e:
        st.error(f"Error loading fallback data: {e}")
        return False, None, None, None, None

def analyze_sentiment(text: str) -> float:
    """Simple sentiment analyzer using keyword matching"""
    if not text or not isinstance(text, str):
        return 0.0
        
    positive_words = {
        'excellent', 'amazing', 'fantastic', 'wonderful', 'great', 'good', 'nice', 
        'delicious', 'tasty', 'fresh', 'clean', 'friendly', 'fast', 'quick',
        'love', 'perfect', 'awesome', 'brilliant', 'outstanding', 'superb',
        'recommend', 'best', 'favorite', 'impressed', 'satisfied', 'enjoy'
    }
    
    negative_words = {
        'terrible', 'awful', 'bad', 'horrible', 'disgusting', 'slow', 'dirty',
        'rude', 'unfriendly', 'cold', 'overpriced', 'expensive', 'small',
        'disappointing', 'worst', 'hate', 'avoid', 'never', 'waste',
        'poor', 'lacking', 'insufficient', 'unacceptable', 'mediocre'
    }
        
    # Clean and tokenize
    words = re.findall(r'\w+', text.lower())
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count + negative_count == 0:
        return 0.0
        
    # Normalize sentiment score
    sentiment = (positive_count - negative_count) / (positive_count + negative_count)
    return sentiment

def get_recommendations(restaurants: pd.DataFrame, ratings: pd.DataFrame, 
                       reviews: pd.DataFrame, num_recommendations: int = 10) -> List[Dict]:
    """Get restaurant recommendations using built-in algorithms"""
    try:
        # Calculate restaurant statistics
        restaurant_stats = ratings.groupby('restaurant_id').agg({
            'rating': ['mean', 'count', 'std']
        }).round(3)
        restaurant_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
        restaurant_stats['rating_std'] = restaurant_stats['rating_std'].fillna(0)
        
        # Calculate sentiment scores for each restaurant
        restaurant_sentiments = {}
        if reviews is not None and 'review_text' in reviews.columns:
            for _, review in reviews.iterrows():
                if pd.notna(review['review_text']):
                    sentiment = analyze_sentiment(review['review_text'])
                    restaurant_id = review['restaurant_id']
                    
                    if restaurant_id not in restaurant_sentiments:
                        restaurant_sentiments[restaurant_id] = []
                    restaurant_sentiments[restaurant_id].append(sentiment)
            
            # Average sentiment per restaurant
            restaurant_sentiments = {
                rid: np.mean(scores) for rid, scores in restaurant_sentiments.items()
            }
        
        # Merge restaurant data with stats
        restaurants_with_stats = restaurants.merge(
            restaurant_stats,
            left_on='restaurant_id',
            right_index=True,
            how='left'
        )
        
        # Fill missing values
        restaurants_with_stats['avg_rating'] = restaurants_with_stats['avg_rating'].fillna(4.0)
        restaurants_with_stats['rating_count'] = restaurants_with_stats['rating_count'].fillna(1)
        
        # Add sentiment scores
        restaurants_with_stats['sentiment_score'] = restaurants_with_stats['restaurant_id'].map(
            lambda x: restaurant_sentiments.get(x, 0.0)
        )
        
        # Calculate popularity score
        restaurants_with_stats['popularity_score'] = (
            restaurants_with_stats['avg_rating'] * 0.4 +
            np.log1p(restaurants_with_stats['rating_count']) * 0.4 +
            (restaurants_with_stats['sentiment_score'] + 1) * 2.5 * 0.2
        )
        
        # Get top recommendations
        top_restaurants = restaurants_with_stats.nlargest(num_recommendations, 'popularity_score')
        
        recommendations = []
        for _, restaurant in top_restaurants.iterrows():
            rec = {
                'restaurant_id': restaurant['restaurant_id'],
                'name': restaurant['name'],
                'cuisine': restaurant['cuisine'],
                'location': restaurant.get('location', 'Unknown'),
                'price_range': restaurant.get('price_range', 'Unknown'),
                'avg_rating': float(restaurant['avg_rating']),
                'rating_count': int(restaurant['rating_count']),
                'sentiment_score': float(restaurant['sentiment_score']),
                'recommendation_score': float(restaurant['popularity_score']),
                'method': 'popularity_based'
            }
            
            # Add additional fields if available
            for field in ['description', 'phone', 'website']:
                if field in restaurant and pd.notna(restaurant[field]):
                    rec[field] = restaurant[field]
            
            recommendations.append(rec)
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

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
    st.markdown("*Production-ready system with built-in algorithms and fallback data*")
    
    # Load fallback data
    data_loaded, restaurants, users, ratings, reviews = load_fallback_data()
    
    if not data_loaded:
        st.error("❌ Unable to load data. Please check your installation.")
        return
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Core Dependencies:**")
            st.write("• Pandas: ✅")
            st.write("• NumPy: ✅")
            st.write("• Streamlit: ✅")
            
        with col2:
            st.write("**Data Status:**")
            st.write(f"• Restaurants: {len(restaurants)}")
            st.write(f"• Users: {len(users)}")
            st.write(f"• Ratings: {len(ratings)}")
            st.write(f"• Reviews: {len(reviews)}")
            
        with col3:
            st.write("**AI Capabilities:**")
            st.write("• Built-in Recommender: ✅")
            st.write("• Sentiment Analysis: ✅")
            st.write("• Popularity-based: ✅")
    
    st.success("✅ AI System Ready!")
    
    # Sidebar for user input
    st.sidebar.header("🎯 Your Preferences")
    
    # Preference filters
    cuisines = ['All'] + list(restaurants['cuisine'].unique())
    selected_cuisine = st.sidebar.selectbox("Preferred Cuisine:", cuisines)
    
    price_ranges = ['All'] + list(restaurants['price_range'].unique())
    selected_price = st.sidebar.selectbox("Price Range:", price_ranges)
    
    min_rating = st.sidebar.slider("Minimum Rating:", 1.0, 5.0, 3.0, 0.1)
    num_recommendations = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)
    
    # Get recommendations
    st.header("🎯 Recommended Restaurants")
    
    with st.spinner("🔍 Generating recommendations..."):
        recommendations = get_recommendations(restaurants, ratings, reviews, num_recommendations * 2)
    
    # Apply filters
    filtered_recommendations = []
    for rec in recommendations:
        # Cuisine filter
        if selected_cuisine != 'All' and rec['cuisine'] != selected_cuisine:
            continue
        
        # Price filter
        if selected_price != 'All' and rec['price_range'] != selected_price:
            continue
        
        # Rating filter
        if rec['avg_rating'] < min_rating:
            continue
        
        filtered_recommendations.append(rec)
    
    # Display recommendations
    if not filtered_recommendations:
        st.warning("🔍 No restaurants match your criteria. Try adjusting your filters.")
    else:
        st.success(f"✨ Found {len(filtered_recommendations)} recommendations!")
        
        # Display recommendations
        for i, rec in enumerate(filtered_recommendations[:num_recommendations], 1):
            with st.expander(
                f"{i}. **{rec['name']}** ⭐ {rec['avg_rating']:.1f} - {rec['cuisine']} - {rec['price_range']}", 
                expanded=i <= 3
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"🍽️ **Cuisine:** {rec['cuisine']}")
                    st.write(f"📍 **Location:** {rec['location']}")
                    st.write(f"💰 **Price Range:** {rec['price_range']}")
                    st.write(f"⭐ **Rating:** {rec['avg_rating']:.1f}/5.0 ({rec['rating_count']} reviews)")
                    
                    if 'sentiment_score' in rec:
                        sentiment = rec['sentiment_score']
                        if sentiment > 0.1:
                            sentiment_emoji = "😊 Positive"
                        elif sentiment < -0.1:
                            sentiment_emoji = "😞 Negative"  
                        else:
                            sentiment_emoji = "😐 Neutral"
                        st.write(f"💭 **Sentiment:** {sentiment_emoji} ({sentiment:.2f})")
                    
                    st.write(f"🤖 **Method:** {rec['method'].replace('_', ' ').title()}")
                    
                    # Additional info
                    if 'description' in rec:
                        st.write(f"📝 **Description:** {rec['description']}")
                    if 'phone' in rec:
                        st.write(f"📞 **Phone:** {rec['phone']}")
                    if 'website' in rec:
                        st.write(f"🌐 **Website:** {rec['website']}")
                
                with col2:
                    score = rec.get('recommendation_score', 0)
                    st.metric("🎯 Recommendation Score", f"{score:.2f}")
                    
                    # Progress bar for score
                    max_score = max([r.get('recommendation_score', 0) for r in filtered_recommendations])
                    if max_score > 0:
                        progress = min(score / max_score, 1.0)
                        st.progress(progress)
                    
                    # Rating breakdown
                    if rec['rating_count'] > 0:
                        st.write("📊 **Quick Stats:**")
                        st.write(f"Reviews: {rec['rating_count']}")
                        
                        # Star rating visualization
                        full_stars = int(rec['avg_rating'])
                        half_star = 1 if rec['avg_rating'] - full_stars >= 0.5 else 0
                        empty_stars = 5 - full_stars - half_star
                        
                        star_display = "⭐" * full_stars + "⭐" * half_star + "☆" * empty_stars
                        st.write(f"Rating: {star_display}")
    
    # Analytics
    st.header("📊 System Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Key Metrics")
        st.metric("Total Restaurants", len(restaurants))
        st.metric("Total Ratings", len(ratings))
        st.metric("Average Rating", f"{ratings['rating'].mean():.2f}")
    
    with col2:
        st.subheader("🍽️ Cuisine Distribution")
        cuisine_counts = restaurants['cuisine'].value_counts()
        
        cuisine_data = pd.DataFrame({
            'Count': cuisine_counts.values
        }, index=cuisine_counts.index)
        st.bar_chart(cuisine_data)
    
    with col3:
        st.subheader("💰 Price Range Distribution")
        price_counts = restaurants['price_range'].value_counts()
        
        price_data = pd.DataFrame({
            'Count': price_counts.values
        }, index=price_counts.index)
        st.bar_chart(price_data)
    
    # Rating distribution
    st.subheader("⭐ Rating Distribution")
    rating_counts = ratings['rating'].value_counts().sort_index()
    rating_data = pd.DataFrame({
        'Count': rating_counts.values
    }, index=rating_counts.index)
    st.bar_chart(rating_data)
    
    # Additional insights
    st.header("🔍 System Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Cuisines")
        top_cuisines = restaurants['cuisine'].value_counts().head(5)
        for cuisine, count in top_cuisines.items():
            st.write(f"• **{cuisine}:** {count} restaurants")
    
    with col2:
        st.subheader("⭐ Highest Rated")
        top_rated = ratings.groupby('restaurant_id')['rating'].mean().nlargest(5)
        for rest_id, rating in top_rated.items():
            try:
                name = restaurants[restaurants['restaurant_id'] == rest_id]['name'].iloc[0]
                st.write(f"• **{name}:** {rating:.1f}⭐")
            except:
                st.write(f"• Restaurant {rest_id}: {rating:.1f}⭐")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 AI Restaurant Recommendation System - Production Ready
    
    **System Features:**
    - 🤖 **Built-in AI Algorithms** - No external ML dependencies
    - 🎯 **Smart Recommendations** - Popularity + sentiment analysis
    - 💭 **Sentiment Analysis** - Keyword-based review analysis
    - 📊 **Real-time Analytics** - Interactive data visualizations
    - 🌐 **Web Interface** - Responsive Streamlit application
    - ☁️ **Cloud Ready** - Optimized for deployment with fallback data
    
    **Technical Stack:**
    - Frontend: Streamlit
    - Backend: Python with built-in algorithms
    - Data: Curated fallback datasets (200 restaurants, 100 users, 1000 ratings)
    - Deployment: Render-optimized with zero external ML dependencies
    
    *Built for production deployment - fast, reliable, and self-contained!*
    """)

if __name__ == "__main__":
    main()
