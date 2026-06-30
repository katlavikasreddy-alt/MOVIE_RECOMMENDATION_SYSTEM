import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Configure Page setup before any imports
st.set_page_config(
    page_title="Cinematique - Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(str(Path(__file__).resolve().parent))
import config
from utils.preprocessing import load_and_preprocess_all
from utils.helpers import (
    get_user_ratings, add_user_rating, delete_user_rating,
    get_watchlist, add_to_watchlist, remove_from_watchlist,
    add_search_query, get_search_history, clear_search_history,
    add_user_review, get_user_reviews, get_all_user_ratings_as_df
)
from utils.recommendation import RecommendationManager, analyze_sentiment
from utils.visualization import (
    plot_genre_distribution, plot_rating_distribution,
    plot_top_rated_movies, plot_popular_movies,
    plot_genre_cooccurrence, plot_movies_by_year
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Premium CSS ---
st.markdown("""
<style>
    /* Main layout improvements */
    .main {
        background: #0d0f12;
        color: #f5f6f8;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #8A1A1A 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(255, 75, 75, 0.2);
    }
    
    .app-header h1 {
        color: white !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        color: #ffe5e5 !important;
        font-size: 1.2rem !important;
        font-weight: 300;
    }
    
    /* Card design */
    .movie-card {
        background: rgba(25, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.4);
        box-shadow: 0 8px 30px rgba(255, 75, 75, 0.15);
    }
    
    .movie-title {
        color: #ff4b4b;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .movie-meta {
        font-size: 0.9rem;
        color: #8f9aa8;
        margin-bottom: 0.8rem;
    }
    
    .metric-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .score-badge {
        background: rgba(0, 204, 150, 0.15);
        color: #00cc96;
        border: 1px solid rgba(0, 204, 150, 0.3);
    }
    
    .reason-text {
        font-style: italic;
        color: #cbd5e1;
        margin-top: 0.8rem;
        padding-left: 0.5rem;
        border-left: 3px solid #ff4b4b;
    }

    /* Glassmorphic sections */
    .glass-panel {
        background: rgba(18, 22, 33, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Global Caching for Data and Models ---
@st.cache_resource
def load_data_and_initialize_manager():
    """
    Loads data and fits models. Caches this resource across sessions.
    """
    with st.spinner("🚀 Bootstrapping Cinematique Engine & Downloading Dataset..."):
        try:
            movies_df, ratings_df, tags_df = load_and_preprocess_all()
            manager = RecommendationManager(collab_kind="item")
            
            # Combine static ratings with any existing sqlite ratings for initialization
            sqlite_ratings = get_all_user_ratings_as_df()
            if not sqlite_ratings.empty:
                full_ratings = pd.concat([ratings_df, sqlite_ratings], ignore_index=True)
            else:
                full_ratings = ratings_df
                
            manager.train_models(movies_df, full_ratings)
            return movies_df, ratings_df, tags_df, manager
        except Exception as e:
            st.error(f"Initialization Failed: {e}")
            logger.error(f"Error bootrapping app: {e}", exc_info=True)
            raise e

# Initialize application data
movies_df, ratings_df, tags_df, rec_manager = load_data_and_initialize_manager()

# --- Syncing Database User Profile ---
# Local user ID is defined in config
local_user_id = config.USER_LOCAL_ID
user_ratings = get_user_ratings(local_user_id)
watchlist = get_watchlist(local_user_id)

# --- Sidebar Navigation ---
st.sidebar.markdown("<h2 style='text-align: center; color: #ff4b4b;'>🎬 Cinematique</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #8f9aa8;'>Movie Recommendation Dashboard</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Menu",
    [
        "🏠 Home & Search",
        "🔍 Get Recommendations",
        "📊 Analytics Dashboard",
        "💬 Reviews & Sentiment",
        "👤 My Profile & History"
    ]
)

st.sidebar.markdown("---")
# Quick stats in sidebar
st.sidebar.markdown("### My Statistics")
st.sidebar.metric("Movies Rated", len(user_ratings))
st.sidebar.metric("Saved Watchlist", len(watchlist))

# --- HEADER HEADER ---
st.markdown("""
<div class="app-header">
    <h1>Cinematique</h1>
    <p>AI-Powered Hybrid Movie Recommendation Engine & Analytics</p>
</div>
""", unsafe_allow_html=True)

# ----------------- PAGE 1: HOME & SEARCH -----------------
if page == "🏠 Home & Search":
    st.subheader("Welcome to Cinematique!")
    
    st.markdown("""
    Cinematique uses advanced **Machine Learning** to recommend movies tailored specifically to your taste. 
    By blending **Content-Based Filtering** (analyzing movie genres and tags) with **Collaborative Filtering** 
    (analyzing similar user rating behaviors), Cinematique offers a robust, hybrid recommendation pipeline.
    
    Explore movies below, submit ratings, and check out the recommendations tab to find your next favorite movie!
    """)
    
    st.markdown("---")
    
    st.subheader("🔍 Movie Directory & Search")
    
    # Search layout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("Search Movie by Title", placeholder="e.g. Toy Story, Inception, Matrix")
    with col2:
        selected_genre = st.selectbox(
            "Filter by Genre",
            ["All"] + sorted(list(set(g for genres in movies_df['genres'].str.split('|') for g in genres if g != '(no genres listed)')))
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Popularity", "Average Rating", "Release Year"]
        )

    # Filter movies list based on user selections
    filtered_df = movies_df.copy()
    
    # Track search query in SQLite
    if search_query.strip():
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False) | 
                                   filtered_df['clean_title'].str.contains(search_query, case=False, na=False)]
        
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, case=False, na=False)]
        
    # Get ratings metadata for sorting
    stats = ratings_df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Merge stats into filtered movie df
    filtered_df = pd.merge(filtered_df, stats, on='movieId', how='left')
    filtered_df['avg_rating'] = filtered_df['avg_rating'].fillna(0.0)
    filtered_df['rating_count'] = filtered_df['rating_count'].fillna(0)
    
    # Add custom ratings into calculation
    for m_id, r in user_ratings.items():
        if m_id in filtered_df['movieId'].values:
            idx = filtered_df[filtered_df['movieId'] == m_id].index[0]
            # Simple weighted update of avg rating for search listing
            current_count = filtered_df.loc[idx, 'rating_count']
            current_avg = filtered_df.loc[idx, 'avg_rating']
            filtered_df.loc[idx, 'rating_count'] = current_count + 1
            filtered_df.loc[idx, 'avg_rating'] = ((current_avg * current_count) + r) / (current_count + 1)
            
    # Sort data
    if sort_by == "Popularity":
        filtered_df = filtered_df.sort_values(by='rating_count', ascending=False)
    elif sort_by == "Average Rating":
        filtered_df = filtered_df.sort_values(by='avg_rating', ascending=False)
    elif sort_by == "Release Year":
        filtered_df = filtered_df.sort_values(by='year', ascending=False)
        
    # Search submission trigger search history logs
    if search_query.strip() and st.session_state.get('last_search') != search_query:
        add_search_query(search_query)
        st.session_state['last_search'] = search_query
        
    st.write(f"Showing {min(50, len(filtered_df))} of {len(filtered_df)} matching movies:")
    
    # Render search results as custom cards
    for idx, row in filtered_df.head(50).iterrows():
        movie_id = int(row['movieId'])
        
        # Display Card
        with st.container():
            col_info, col_actions = st.columns([3, 1])
            with col_info:
                st.markdown(f"<div class='movie-title'>{row['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='movie-meta'>
                    <span class='metric-badge'>📅 Year: {int(row['year']) if row['year'] > 0 else 'Unknown'}</span>
                    <span class='metric-badge'>🏷️ Genres: {row['genres'].replace('|', ', ')}</span>
                    <span class='metric-badge'>⭐ Avg Rating: {row['avg_rating']:.2f} ({int(row['rating_count'])} ratings)</span>
                </div>
                """, unsafe_allow_html=True)
                
            with col_actions:
                # Add to watchlist & Rate button
                in_watchlist = movie_id in watchlist
                watchlist_btn_txt = "🔖 Save Movie" if not in_watchlist else "✅ In Watchlist"
                
                # Check current user rating
                curr_rating = user_ratings.get(movie_id, 0.0)
                
                c_btn, c_rate = st.columns([1, 1])
                with c_btn:
                    if st.button(watchlist_btn_txt, key=f"wl_{movie_id}"):
                        if in_watchlist:
                            remove_from_watchlist(local_user_id, movie_id)
                            st.rerun()
                        else:
                            add_to_watchlist(local_user_id, movie_id)
                            st.success(f"Added {row['clean_title']} to Watchlist!")
                            st.rerun()
                            
                with c_rate:
                    # Choose new rating
                    rating_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                    default_idx = rating_options.index(curr_rating) if curr_rating in rating_options else 0
                    
                    selected_rating = st.selectbox(
                        "Rate",
                        rating_options,
                        index=default_idx,
                        key=f"rate_sel_{movie_id}"
                    )
                    
                    # Update database if selection differs
                    if selected_rating != curr_rating:
                        if selected_rating == 0.0:
                            delete_user_rating(local_user_id, movie_id)
                        else:
                            add_user_rating(local_user_id, movie_id, selected_rating)
                        st.rerun()
            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.15;'/>", unsafe_allow_html=True)

# ----------------- PAGE 2: RECOMMENDATION ENGINE -----------------
elif page == "🔍 Get Recommendations":
    st.subheader("🤖 Recommendation Engine Panel")
    
    # Warning if user has no ratings
    if len(user_ratings) == 0:
        st.warning("⚠️ You haven't rated any movies yet! The algorithms will fall back to general popularity metrics. Rate a few movies on the 'Home' tab to generate personalized recommendations!")
        
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rec_method = st.selectbox(
            "Select Recommendation Algorithm",
            [
                "Hybrid (Content + Collaborative)",
                "Content-Based Filtering (Genre & Tag overlap)",
                "Collaborative Filtering (Ratings-based similar users/items)"
            ]
        )
        
        # Select sub-method for collaborative
        if "Collaborative" in rec_method or "Hybrid" in rec_method:
            collab_type = st.radio("Collaborative Approach", ["Item-Based (Similar Movies)", "User-Based (Similar Users)", "Matrix Factorization (SVD)"])
            # Update recommender configuration if changed
            if collab_type == "Item-Based (Similar Movies)" and rec_manager.hybrid_engine.collab_model.kind != "item":
                rec_manager.hybrid_engine.collab_model = RecommendationManager(collab_kind="item").hybrid_engine.collab_model
                rec_manager.train_models(movies_df, pd.concat([ratings_df, get_all_user_ratings_as_df()], ignore_index=True))
            elif collab_type == "User-Based (Similar Users)" and rec_manager.hybrid_engine.collab_model.kind != "user":
                rec_manager.hybrid_engine.collab_model = RecommendationManager(collab_kind="user").hybrid_engine.collab_model
                rec_manager.train_models(movies_df, pd.concat([ratings_df, get_all_user_ratings_as_df()], ignore_index=True))
            elif collab_type == "Matrix Factorization (SVD)" and rec_manager.hybrid_engine.collab_model.kind != "svd":
                rec_manager.hybrid_engine.collab_model = RecommendationManager(collab_kind="svd").hybrid_engine.collab_model
                rec_manager.train_models(movies_df, pd.concat([ratings_df, get_all_user_ratings_as_df()], ignore_index=True))
                
        num_recs = st.slider("Number of Recommendations", min_value=1, max_value=20, value=config.DEFAULT_NUM_RECOMMENDATIONS)
        
    with col2:
        # Display Hybrid Weights sliders
        if "Hybrid" in rec_method:
            st.write("##### Set Hybrid Blending Ratio")
            content_weight = st.slider("Content Weight (Genres/Tags)", 0.0, 1.0, config.HYBRID_CONTENT_WEIGHT, 0.05)
            collab_weight = 1.0 - content_weight
            st.info(f"⚖️ Hybrid ratio set: {content_weight*100:.0f}% Content / {collab_weight*100:.0f}% Collaborative")
        else:
            content_weight, collab_weight = 0.5, 0.5
            
        enable_sentiment = st.checkbox("Enable Sentiment-Based Boosting", value=True, help="Adjusts recommendations score dynamically based on positive or negative review sentiments in the database.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendation trigger button
    if st.button("🎬 Generate Recommendations", type="primary"):
        # Map selected label to keyword argument
        method_kw = "hybrid"
        if "Content-Based" in rec_method:
            method_kw = "content"
        elif "Collaborative" in rec_method:
            method_kw = "collab"
            
        # Retrain CF matrix dynamically on active ratings in SQLite database before running recommendations
        # This solves the requirement "Dynamic Recommendation Updates"
        sqlite_ratings = get_all_user_ratings_as_df()
        full_ratings = pd.concat([ratings_df, sqlite_ratings], ignore_index=True)
        rec_manager.train_models(movies_df, full_ratings)
            
        with st.spinner("🔮 Analyzing similarity matrices and calculating recommendations..."):
            try:
                recs = rec_manager.get_recommendations(
                    user_ratings_dict=user_ratings,
                    method=method_kw,
                    w_content=content_weight,
                    w_collab=collab_weight,
                    enable_sentiment=enable_sentiment,
                    top_n=num_recs
                )
                
                if not recs:
                    st.error("Could not generate recommendations for the given profile settings.")
                else:
                    st.success(f"Successfully generated top {len(recs)} movie recommendations!")
                    st.markdown("---")
                    
                    # Display recommendations list
                    for rec in recs:
                        movie_id = rec['movieId']
                        in_wl = movie_id in watchlist
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">#{rec['rank']} - {rec['title']}</div>
                            <div class="movie-meta">
                                <span class="metric-badge">📅 Year: {rec['year']}</span>
                                <span class="metric-badge">🏷️ Genres: {rec['genres'].replace('|', ', ')}</span>
                                <span class="metric-badge">⭐ Predicted Rating: {rec['predicted_rating']:.2f}/5.0</span>
                                <span class="metric-badge score-badge">🎯 Match Score: {rec['hybrid_score']*100:.1f}%</span>
                                <span class="metric-badge">📊 Popularity Count: {rec['popularity_score']} ratings</span>
                            </div>
                            <div class="reason-text">💡 <strong>Recommendation Reason:</strong> {rec['reason']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Actions for recommended items
                        col1, col2 = st.columns([1, 6])
                        with col1:
                            # Quick Watchlist save button
                            btn_label = "🔖 Watchlist" if not in_wl else "✅ Watchlist"
                            if st.button(btn_label, key=f"rec_wl_{movie_id}"):
                                if in_wl:
                                    remove_from_watchlist(local_user_id, movie_id)
                                else:
                                    add_to_watchlist(local_user_id, movie_id)
                                st.rerun()
                        with col2:
                            # Give ratings quick-change
                            curr_val = user_ratings.get(movie_id, 0.0)
                            rating_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                            def_idx = rating_options.index(curr_val) if curr_val in rating_options else 0
                            
                            new_r = st.selectbox(
                                "Rate this movie",
                                rating_options,
                                index=def_idx,
                                key=f"rec_rate_{movie_id}",
                                label_visibility="collapsed"
                            )
                            if new_r != curr_val:
                                if new_r == 0.0:
                                    delete_user_rating(local_user_id, movie_id)
                                else:
                                    add_user_rating(local_user_id, movie_id, new_r)
                                st.rerun()
                                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                logger.error(f"Error recommendation run: {e}", exc_info=True)

# ----------------- PAGE 3: ANALYTICS DASHBOARD -----------------
elif page == "📊 Analytics Dashboard":
    st.subheader("📊 Movie Lens Dataset Insights & User Dashboard")
    
    # Layout KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Movies in DB", len(movies_df))
    with col2:
        # User base count
        num_users = ratings_df['userId'].nunique()
        st.metric("Total Users in DB", num_users)
    with col3:
        st.metric("Total Ratings", len(ratings_df))
    with col4:
        st.metric("My Custom Ratings", len(user_ratings))
    with col5:
        st.metric("Watchlist Saved", len(watchlist))
        
    st.markdown("---")
    
    # Dashboard visualizer tabs
    tab1, tab2, tab3 = st.tabs(["🍿 Movies & Genres", "⭐ Ratings & Trends", "🔥 Popular & Top Rated"])
    
    with tab1:
        st.write("##### Genre Distributions and Releases")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(plot_genre_distribution(movies_df), use_container_width=True)
        with col_g2:
            st.plotly_chart(plot_movies_by_year(movies_df), use_container_width=True)
            
        st.write("##### Genre Co-Occurrence Network Heatmap")
        st.plotly_chart(plot_genre_cooccurrence(movies_df), use_container_width=True)
        
    with tab2:
        st.write("##### Ratings Distribution Analysis")
        st.plotly_chart(plot_rating_distribution(ratings_df), use_container_width=True)
        
    with tab3:
        st.write("##### Charts of Popular & Top Rated Titles")
        col_pop, col_rate = st.columns(2)
        with col_pop:
            st.plotly_chart(plot_popular_movies(movies_df, ratings_df, top_n=10), use_container_width=True)
        with col_rate:
            st.plotly_chart(plot_top_rated_movies(movies_df, ratings_df, min_ratings=50, top_n=10), use_container_width=True)

# ----------------- PAGE 4: REVIEWS & SENTIMENT -----------------
elif page == "💬 Reviews & Sentiment":
    st.subheader("💬 User Reviews & Sentiment Analysis Dashboard")
    
    st.markdown("""
    Cinematique allows you to write review texts for movies. The system processes your text through a **Sentiment Analysis** 
    algorithm to automatically classify it as positive, neutral, or negative.
    
    These reviews dynamically impact movie recommendations by boosting or penalizing the calculated prediction rankings!
    """)
    
    st.markdown("---")
    
    col_write, col_reviews = st.columns([1, 1])
    
    with col_write:
        st.write("#### Write a Movie Review")
        
        # Movie selection dropdown
        movie_titles_mapping = {row['title']: row['movieId'] for _, row in movies_df.iterrows()}
        selected_title = st.selectbox(
            "Select Movie to Review",
            options=sorted(list(movie_titles_mapping.keys()))
        )
        
        review_movie_id = movie_titles_mapping[selected_title]
        review_text = st.text_area("Write your review text", height=150, placeholder="Type your thoughts here... e.g. This was an amazing classic masterpiece! I loved the performances.")
        
        if st.button("Submit Review"):
            if not review_text.strip():
                st.error("Please enter a review text.")
            else:
                # Perform Sentiment Analysis
                sentiment_label, score = analyze_sentiment(review_text)
                
                # Save to database
                add_user_review(local_user_id, review_movie_id, review_text, sentiment_label, score)
                
                # Feedback to user
                emoji = "🟢" if sentiment_label == "Positive" else "🔴" if sentiment_label == "Negative" else "🟡"
                st.success(f"Review Submitted successfully! Sentiment analyzed as: {emoji} **{sentiment_label}** (score: {score:+.2f})")
                st.rerun()
                
    with col_reviews:
        st.write("#### Existing Reviews in DB")
        
        # Display submitted reviews
        all_reviews = get_user_reviews()
        if not all_reviews:
            st.info("No user reviews added yet. Submit your first review!")
        else:
            for review in all_reviews:
                movie_name = movies_df[movies_df['movieId'] == review['movieId']]['title'].values[0]
                label = review['sentiment_label']
                emoji = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "🟡"
                
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem;">
                    <div style="font-weight: bold; color: #ff4b4b; display: flex; justify-content: space-between;">
                        <span>{movie_name}</span>
                        <span>{emoji} {label}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #8f9aa8; margin-bottom: 0.5rem;">Reviewed on {review['reviewed_at']}</div>
                    <p style="font-size: 0.95rem; margin: 0; color: #cbd5e1;">"{review['review_text']}"</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 5: MY PROFILE -----------------
elif page == "👤 My Profile & History":
    st.subheader("👤 User Profile & Activity Logs")
    
    tab_rat, tab_wl, tab_sh = st.tabs(["⭐ My Ratings", "🔖 My Watchlist", "🕵️ Search History"])
    
    with tab_rat:
        st.write("##### Movies You've Rated")
        if not user_ratings:
            st.info("You haven't rated any movies yet. Go to Home or Recommendations page to start rating!")
        else:
            ratings_data = []
            for m_id, r in user_ratings.items():
                movie_title = movies_df[movies_df['movieId'] == m_id]['title'].values[0]
                movie_genre = movies_df[movies_df['movieId'] == m_id]['genres'].values[0]
                ratings_data.append({'Movie ID': m_id, 'Title': movie_title, 'Genre': movie_genre, 'My Rating': r})
                
            rat_df = pd.DataFrame(ratings_data)
            
            # Interactive delete option in profile
            for idx, row in rat_df.iterrows():
                col_name, col_rating, col_del = st.columns([4, 1, 1])
                with col_name:
                    st.write(f"**{row['Title']}** ({row['Genre'].replace('|', ', ')})")
                with col_rating:
                    st.write(f"⭐ {row['My Rating']}/5.0")
                with col_del:
                    if st.button("Delete Rating", key=f"del_r_{row['Movie ID']}"):
                        delete_user_rating(local_user_id, int(row['Movie ID']))
                        st.rerun()
                st.markdown("<hr style='margin: 0.3rem 0; opacity: 0.1;'/>", unsafe_allow_html=True)

    with tab_wl:
        st.write("##### Movies Saved in Your Watchlist")
        if not watchlist:
            st.info("Your watchlist is empty. Search movies on the Home page to bookmark them!")
        else:
            for m_id in watchlist:
                title = movies_df[movies_df['movieId'] == m_id]['title'].values[0]
                genres = movies_df[movies_df['movieId'] == m_id]['genres'].values[0]
                
                col_title, col_rem = st.columns([5, 1])
                with col_title:
                    st.write(f"🔖 **{title}** ({genres.replace('|', ', ')})")
                with col_rem:
                    if st.button("Remove", key=f"rem_wl_p_{m_id}"):
                        remove_from_watchlist(local_user_id, m_id)
                        st.rerun()
                st.markdown("<hr style='margin: 0.3rem 0; opacity: 0.1;'/>", unsafe_allow_html=True)
                
    with tab_sh:
        st.write("##### Your Search History")
        history = get_search_history(15)
        
        if not history:
            st.info("No search history recorded.")
        else:
            if st.button("Clear Search History"):
                clear_search_history()
                st.rerun()
                
            for idx, h in enumerate(history):
                st.write(f"{idx+1}. **\"{h['query']}\"** — searched on {h['searched_at']}")
                
    # Database reset button
    st.markdown("---")
    st.write("##### Danger Zone")
    if st.button("Reset All Application Cache & DB Tables"):
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS user_ratings")
        cursor.execute("DROP TABLE IF EXISTS watchlist")
        cursor.execute("DROP TABLE IF EXISTS search_history")
        cursor.execute("DROP TABLE IF EXISTS user_reviews")
        cursor.execute("DROP TABLE IF EXISTS recommendation_history")
        conn.commit()
        conn.close()
        st.cache_resource.clear()
        st.success("Successfully reset database tables! Refreshing application...")
        st.rerun()
