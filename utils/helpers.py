import sqlite3
import pandas as pd
import datetime
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(config.DB_PATH)
    # Return rows as dictionary-like objects
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database tables.
    """
    logger.info("Initializing SQLite database tables...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. User ratings (for local custom user)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_ratings (
        userId INTEGER,
        movieId INTEGER,
        rating REAL,
        timestamp INTEGER,
        PRIMARY KEY (userId, movieId)
    )
    """)
    
    # 2. Watchlist / Saved movies
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        userId INTEGER,
        movieId INTEGER,
        added_at TEXT,
        PRIMARY KEY (userId, movieId)
    )
    """)
    
    # 3. Search history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        searched_at TEXT
    )
    """)
    
    # 4. User reviews (for sentiment analysis)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        userId INTEGER,
        movieId INTEGER,
        review_text TEXT,
        sentiment_label TEXT,
        sentiment_score REAL,
        reviewed_at TEXT
    )
    """)
    
    # 5. Recommendation history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        userId INTEGER,
        rec_type TEXT,
        recommended_movie_ids TEXT,
        created_at TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# --- Rating Functions ---
def add_user_rating(user_id: int, movie_id: int, rating: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = int(datetime.datetime.now().timestamp())
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO user_ratings (userId, movieId, rating, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, movie_id, rating, timestamp)
        )
        conn.commit()
        logger.info(f"Added rating of {rating} for movieId {movie_id} by userId {user_id}")
    except Exception as e:
        logger.error(f"Error adding rating: {e}")
        conn.rollback()
    finally:
        conn.close()

def delete_user_rating(user_id: int, movie_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM user_ratings WHERE userId = ? AND movieId = ?", (user_id, movie_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error deleting rating: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_user_ratings(user_id: int) -> dict[int, float]:
    """
    Returns user ratings as a dictionary: {movieId: rating}
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    ratings = {}
    try:
        cursor.execute("SELECT movieId, rating FROM user_ratings WHERE userId = ?", (user_id,))
        for row in cursor.fetchall():
            ratings[row['movieId']] = row['rating']
    except Exception as e:
        logger.error(f"Error getting user ratings: {e}")
    finally:
        conn.close()
    return ratings

def get_user_ratings_as_df(user_id: int) -> pd.DataFrame:
    """
    Returns user ratings as a Pandas DataFrame matching ratings.csv.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT userId, movieId, rating, timestamp FROM user_ratings WHERE userId = ?", conn, params=(user_id,))
    except Exception as e:
        logger.error(f"Error getting user ratings as df: {e}")
        df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    finally:
        conn.close()
    return df

def get_all_user_ratings_as_df() -> pd.DataFrame:
    """
    Returns ALL custom ratings in SQLite as a Pandas DataFrame.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT userId, movieId, rating, timestamp FROM user_ratings", conn)
    except Exception as e:
        logger.error(f"Error getting all ratings from DB: {e}")
        df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    finally:
        conn.close()
    return df

# --- Watchlist Functions ---
def add_to_watchlist(user_id: int, movie_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    added_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO watchlist (userId, movieId, added_at) VALUES (?, ?, ?)",
            (user_id, movie_id, added_at)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        conn.rollback()
    finally:
        conn.close()

def remove_from_watchlist(user_id: int, movie_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM watchlist WHERE userId = ? AND movieId = ?", (user_id, movie_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_watchlist(user_id: int) -> list[int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    movie_ids = []
    try:
        cursor.execute("SELECT movieId FROM watchlist WHERE userId = ?", (user_id,))
        movie_ids = [row['movieId'] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error reading watchlist: {e}")
    finally:
        conn.close()
    return movie_ids

# --- Search History Functions ---
def add_search_query(query: str):
    if not query.strip():
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    searched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("INSERT INTO search_history (query, searched_at) VALUES (?, ?)", (query.strip(), searched_at))
        conn.commit()
    except Exception as e:
        logger.error(f"Error adding search: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_search_history(limit: int = 10) -> list[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    history = []
    try:
        cursor.execute("SELECT query, searched_at FROM search_history ORDER BY id DESC LIMIT ?", (limit,))
        history = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error reading search history: {e}")
    finally:
        conn.close()
    return history

def clear_search_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM search_history")
        conn.commit()
    except Exception as e:
        logger.error(f"Error clearing search: {e}")
        conn.rollback()
    finally:
        conn.close()

# --- Reviews and Sentiment Functions ---
def add_user_review(user_id: int, movie_id: int, review_text: str, sentiment_label: str, sentiment_score: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    reviewed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute(
            "INSERT INTO user_reviews (userId, movieId, review_text, sentiment_label, sentiment_score, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, movie_id, review_text, sentiment_label, sentiment_score, reviewed_at)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error adding review: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_user_reviews(movie_id: int = None) -> list[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    reviews = []
    try:
        if movie_id:
            cursor.execute("SELECT userId, movieId, review_text, sentiment_label, sentiment_score, reviewed_at FROM user_reviews WHERE movieId = ? ORDER BY id DESC", (movie_id,))
        else:
            cursor.execute("SELECT userId, movieId, review_text, sentiment_label, sentiment_score, reviewed_at FROM user_reviews ORDER BY id DESC")
        reviews = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error reading reviews: {e}")
    finally:
        conn.close()
    return reviews

def get_average_sentiment_boost(movie_id: int) -> float:
    """
    Returns average sentiment score of reviews for a movie.
    Sentiment score is normalized between -1.0 and 1.0.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    avg_score = 0.0
    try:
        cursor.execute("SELECT AVG(sentiment_score) as avg_score FROM user_reviews WHERE movieId = ?", (movie_id,))
        row = cursor.fetchone()
        if row and row['avg_score'] is not None:
            avg_score = float(row['avg_score'])
    except Exception as e:
        logger.error(f"Error calculating average sentiment: {e}")
    finally:
        conn.close()
    return avg_score

# --- Recommendation History ---
def add_recommendation_history(user_id: int, rec_type: str, recommended_ids: list[int]):
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids_str = ",".join(map(str, recommended_ids))
    try:
        cursor.execute(
            "INSERT INTO recommendation_history (userId, rec_type, recommended_movie_ids, created_at) VALUES (?, ?, ?, ?)",
            (user_id, rec_type, ids_str, created_at)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error writing recommendation history: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_recommendation_history(user_id: int, limit: int = 5) -> list[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    history = []
    try:
        cursor.execute("SELECT rec_type, recommended_movie_ids, created_at FROM recommendation_history WHERE userId = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
        for row in cursor.fetchall():
            history.append({
                'rec_type': row['rec_type'],
                'movie_ids': [int(x) for x in row['recommended_movie_ids'].split(",") if x],
                'created_at': row['created_at']
            })
    except Exception as e:
        logger.error(f"Error reading recommendation history: {e}")
    finally:
        conn.close()
    return history

# Initialize SQLite database immediately upon import
init_db()
