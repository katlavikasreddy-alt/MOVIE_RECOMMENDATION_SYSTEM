import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import sqlite3

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.preprocessing import clean_title, extract_year
from utils.helpers import get_user_ratings, add_user_rating, delete_user_rating, init_db
from utils.recommendation import analyze_sentiment
from models.content_filter import ContentRecommender
from models.collaborative_filter import CollaborativeRecommender
from models.hybrid_model import HybridRecommender

# Sample data for unit testing models
@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4],
        'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'Sabrina (1995)'],
        'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 'Action|Crime|Thriller', 'Comedy|Romance'],
        'genres_processed': ['Adventure Animation Children Comedy Fantasy', 'Adventure Children Fantasy', 'Action Crime Thriller', 'Comedy Romance'],
        'year': [1995, 1995, 1995, 1995],
        'clean_title': ['Toy Story', 'Jumanji', 'Heat', 'Sabrina'],
        'metadata': [
            'Adventure Animation Children Comedy Fantasy toy-story clever pixar',
            'Adventure Children Fantasy board game magic',
            'Action Crime Thriller robbery gun heist robbery',
            'Comedy Romance remake modern love'
        ]
    })

@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        'userId': [1, 1, 2, 2, 3, 3, 4, 4],
        'movieId': [1, 2, 2, 3, 1, 3, 1, 4],
        'rating': [5.0, 3.0, 4.0, 5.0, 2.0, 4.0, 4.0, 5.0],
        'timestamp': [964982703, 964981247, 964982224, 964983815, 964981700, 964982500, 964981100, 964983100]
    })

# --- Helper function tests ---
def test_clean_title():
    assert clean_title("Toy Story (1995)") == "Toy Story"
    assert clean_title("GoldenEye (1995)") == "GoldenEye"
    assert clean_title("The Matrix") == "The Matrix"
    assert clean_title(None) == ""

def test_extract_year():
    assert extract_year("Toy Story (1995)") == 1995
    assert extract_year("GoldenEye (1995)") == 1995
    assert extract_year("The Matrix") == 0
    assert extract_year(None) == 0

# --- Sentiment Analysis tests ---
def test_analyze_sentiment():
    # Test positive reviews
    label, score = analyze_sentiment("This movie was great and highly entertaining!")
    assert label == "Positive"
    assert score > 0
    
    # Test negative reviews
    label, score = analyze_sentiment("This was a boring terrible waste of time.")
    assert label == "Negative"
    assert score < 0
    
    # Test negation handling
    label_neg, score_neg = analyze_sentiment("This is not a bad movie.")
    assert label_neg == "Positive" # 'not bad' should result in positive
    
    label_neg2, score_neg2 = analyze_sentiment("The story was not good.")
    assert label_neg2 == "Negative" # 'not good' should result in negative
    
    # Test neutral / empty case
    label_neu, score_neu = analyze_sentiment("The movie has some actors and is a film.")
    assert label_neu == "Neutral"
    assert score_neu == 0.0

# --- Model Unit Tests ---
def test_content_recommender(sample_movies):
    recommender = ContentRecommender()
    recommender.fit(sample_movies)
    
    # Check vocabulary mappings
    assert 1 in recommender.movie_id_to_idx
    assert len(recommender.movie_id_to_idx) == 4
    assert recommender.tfidf_matrix.shape == (4, len(recommender.vectorizer.get_feature_names_out()))
    
    # Test item recommendations
    recs = recommender.recommend_similar_movies(1, top_n=2)
    assert len(recs) == 2
    # Toy Story (1) should share similarity with Jumanji (2) due to 'Adventure' and 'Fantasy'
    assert recs[0][0] == 2 # Check that Jumanji is recommended
    assert recs[0][1] > 0

def test_collaborative_recommender(sample_ratings):
    # Test Item-Based CF
    rec_item = CollaborativeRecommender(kind="item")
    rec_item.fit(sample_ratings)
    assert rec_item.user_item_matrix.shape == (4, 4) # 4 users, 4 movies
    
    # Make recommendation for a mock profile
    mock_profile = {1: 5.0, 2: 3.0}
    recs = rec_item.recommend_for_user(mock_profile, top_n=2)
    assert len(recs) > 0

    # Test User-Based CF
    rec_user = CollaborativeRecommender(kind="user")
    rec_user.fit(sample_ratings)
    assert rec_user.user_item_matrix.shape == (4, 4)

    # Test SVD Matrix Factorization CF
    rec_svd = CollaborativeRecommender(kind="svd")
    rec_svd.fit(sample_ratings)
    assert rec_svd.user_item_matrix.shape == (4, 4)
    assert rec_svd.svd_components is not None
    
    recs_svd = rec_svd.recommend_for_user(mock_profile, top_n=2)
    assert len(recs_svd) > 0

def test_hybrid_recommender(sample_movies, sample_ratings):
    hybrid = HybridRecommender(collab_kind="item")
    hybrid.fit(sample_movies, sample_ratings)
    
    mock_profile = {1: 4.5}
    recs = hybrid.recommend_hybrid(mock_profile, w_content=0.5, w_collab=0.5, top_n=2)
    
    assert len(recs) <= 2
    for rec in recs:
        assert 'movieId' in rec
        assert 'title' in rec
        assert 'hybrid_score' in rec
        assert 'reason' in rec

# --- Database Integration Tests ---
def test_database_operations():
    # Re-initialize test database tables
    init_db()
    
    test_user = 88888
    test_movie = 9999
    test_rating = 4.5
    
    # Set custom rating
    add_user_rating(test_user, test_movie, test_rating)
    
    # Fetch ratings
    ratings = get_user_ratings(test_user)
    assert test_movie in ratings
    assert ratings[test_movie] == test_rating
    
    # Delete rating
    delete_user_rating(test_user, test_movie)
    ratings_after = get_user_ratings(test_user)
    assert test_movie not in ratings_after
