import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class ContentRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[\w-]+\b')
        self.tfidf_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, movies_df: pd.DataFrame):
        """
        Fits TF-IDF Vectorizer on movie metadata (processed genres + tags).
        """
        logger.info("Fitting Content-Based recommender model...")
        self.movies_df = movies_df.copy()
        
        # Ensure metadata is string
        metadata_series = self.movies_df['metadata'].fillna('')
        
        # Compute TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(metadata_series)
        
        # Create mapping matrices
        self.movie_id_to_idx = {row['movieId']: idx for idx, row in self.movies_df.iterrows()}
        self.idx_to_movie_id = {idx: row['movieId'] for idx, row in self.movies_df.iterrows()}
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
    def save(self, filepath_prefix: str = "content_model"):
        """
        Saves the fitted model files to config.MODELS_DIR.
        """
        if self.tfidf_matrix is None or self.movies_df is None:
            raise ValueError("Model must be fitted before saving.")
            
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id,
                'movies_df': self.movies_df[['movieId', 'title', 'clean_title', 'genres', 'genres_processed', 'year', 'metadata']]
            }
            path = config.MODELS_DIR / f"{filepath_prefix}.joblib"
            joblib.dump(model_data, path)
            logger.info(f"Content model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving content model: {e}")

    def load(self, filepath_prefix: str = "content_model") -> bool:
        """
        Loads the saved model files from config.MODELS_DIR.
        """
        path = config.MODELS_DIR / f"{filepath_prefix}.joblib"
        if not path.exists():
            logger.warning(f"Saved content model not found at {path}")
            return False
            
        try:
            model_data = joblib.load(path)
            self.vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.movie_id_to_idx = model_data['movie_id_to_idx']
            self.idx_to_movie_id = model_data['idx_to_movie_id']
            self.movies_df = model_data['movies_df']
            logger.info("Content model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading content model: {e}")
            return False

    def get_similarity_scores_for_movie(self, movie_id: int) -> np.ndarray:
        """
        Computes cosine similarities between target movie and all movies in the dataset.
        Returns array of similarity scores.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model is not fitted or loaded.")
            
        if movie_id not in self.movie_id_to_idx:
            logger.warning(f"Movie ID {movie_id} not found in model vocabulary.")
            return np.zeros(self.tfidf_matrix.shape[0])
            
        idx = self.movie_id_to_idx[movie_id]
        movie_vector = self.tfidf_matrix[idx]
        
        # Calculate cosine similarity using linear_kernel (since TF-IDF vectors are L2-normalized)
        sim_scores = linear_kernel(movie_vector, self.tfidf_matrix).flatten()
        return sim_scores

    def recommend_similar_movies(self, movie_id: int, top_n: int = 5) -> list[tuple[int, float]]:
        """
        Recommends top_n movies similar to a given movie.
        Returns list of (movieId, similarity_score).
        """
        sim_scores = self.get_similarity_scores_for_movie(movie_id)
        if np.all(sim_scores == 0):
            return []
            
        # Get sorted index positions (descending order)
        sim_indices = np.argsort(sim_scores)[::-1]
        
        # Filter out the queried movie itself
        target_idx = self.movie_id_to_idx[movie_id]
        sim_indices = [idx for idx in sim_indices if idx != target_idx]
        
        recommendations = []
        for idx in sim_indices[:top_n]:
            rec_movie_id = self.idx_to_movie_id[idx]
            recommendations.append((rec_movie_id, float(sim_scores[idx])))
            
        return recommendations

    def recommend_for_user_profile(self, user_ratings: dict[int, float], top_n: int = 5) -> list[tuple[int, float]]:
        """
        Builds user profile vector from positive user ratings and finds similar movies.
        user_ratings: dict of {movieId: rating}
        Returns list of (movieId, similarity_score).
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model is not fitted or loaded.")
            
        if not user_ratings:
            logger.warning("Empty user ratings profile provided. Cannot compute recommendations.")
            return []
            
        # Standardize ratings by subtracting a baseline (e.g. 2.5) to weight positive ratings positive and negative ratings negative.
        # This prevents disliked movies from contributing positively to recommendations.
        profile_vector = np.zeros(self.tfidf_matrix.shape[1])
        weight_sum = 0.0
        
        for movie_id, rating in user_ratings.items():
            if movie_id not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[movie_id]
            movie_vector = self.tfidf_matrix[idx].toarray().flatten()
            
            # Weight is centered rating
            weight = rating - 2.5
            profile_vector += movie_vector * weight
            weight_sum += abs(weight)
            
        if weight_sum == 0 or np.all(profile_vector == 0):
            # Fallback if no weights or blank profile
            return []
            
        # Normalize the profile vector
        profile_vector = profile_vector / np.linalg.norm(profile_vector)
        
        # Calculate cosine similarity with all movie TF-IDF vectors
        sim_scores = linear_kernel(profile_vector.reshape(1, -1), self.tfidf_matrix).flatten()
        
        # Sort indices descending
        sim_indices = np.argsort(sim_scores)[::-1]
        
        # Filter out movies the user has already rated
        rated_movie_ids = set(user_ratings.keys())
        
        recommendations = []
        for idx in sim_indices:
            rec_movie_id = self.idx_to_movie_id[idx]
            if rec_movie_id not in rated_movie_ids:
                recommendations.append((rec_movie_id, float(sim_scores[idx])))
            if len(recommendations) == top_n:
                break
                
        return recommendations
