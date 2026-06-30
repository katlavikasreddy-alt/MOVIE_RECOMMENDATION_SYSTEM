import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class CollaborativeRecommender:
    def __init__(self, kind: str = "item"):
        """
        kind: "user" for User-Based CF, "item" for Item-Based CF, "svd" for Matrix Factorization.
        """
        if kind not in ["user", "item", "svd"]:
            raise ValueError("Kind must be 'user', 'item', or 'svd'")
        self.kind = kind
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.svd_components = None
        self.reconstructed_matrix = None
        self.user_means = None
        self.movie_means = None
        
        # Mapping helpers
        self.movie_id_to_col_idx = {}
        self.col_idx_to_movie_id = {}
        self.user_id_to_row_idx = {}
        self.row_idx_to_user_id = {}

    def fit(self, ratings_df: pd.DataFrame):
        """
        Fits the collaborative filter model by constructing the user-item matrix
        and calculating similarity or projection matrices.
        """
        logger.info(f"Fitting {self.kind.upper()} Collaborative Filtering model...")
        
        # Create user-item pivot table
        pivot_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        
        # Save mappings
        self.movie_id_to_col_idx = {movie_id: idx for idx, movie_id in enumerate(pivot_df.columns)}
        self.col_idx_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(pivot_df.columns)}
        self.user_id_to_row_idx = {user_id: idx for idx, user_id in enumerate(pivot_df.index)}
        self.row_idx_to_user_id = {idx: user_id for idx, user_id in enumerate(pivot_df.index)}
        
        self.user_means = pivot_df.mean(axis=1).values
        self.movie_means = pivot_df.mean(axis=0).values
        
        # Fill missing ratings with 0
        self.user_item_matrix = pivot_df.fillna(0).values
        
        if self.kind == "user":
            # For User-Based CF: calculate user-user similarity matrix (N x N)
            centered_matrix = np.zeros_like(self.user_item_matrix)
            for i in range(self.user_item_matrix.shape[0]):
                mask = self.user_item_matrix[i] > 0
                if mask.any():
                    centered_matrix[i, mask] = self.user_item_matrix[i, mask] - self.user_means[i]
            
            self.similarity_matrix = cosine_similarity(centered_matrix)
            np.fill_diagonal(self.similarity_matrix, 0)
            logger.info(f"User-user similarity matrix computed: {self.similarity_matrix.shape}")
            
        elif self.kind == "item":
            # For Item-Based CF: calculate item-item similarity matrix (M x M)
            centered_matrix = np.zeros_like(self.user_item_matrix)
            for j in range(self.user_item_matrix.shape[1]):
                mask = self.user_item_matrix[:, j] > 0
                if mask.any():
                    centered_matrix[mask, j] = self.user_item_matrix[mask, j] - self.movie_means[j]
            
            self.similarity_matrix = cosine_similarity(centered_matrix.T)
            np.fill_diagonal(self.similarity_matrix, 0)
            logger.info(f"Item-item similarity matrix computed: {self.similarity_matrix.shape}")
            
        else: # svd
            # Matrix Factorization (SVD)
            logger.info("Decomposing mean-centered ratings matrix using TruncatedSVD...")
            centered_matrix = np.zeros_like(self.user_item_matrix)
            for j in range(self.user_item_matrix.shape[1]):
                mask = self.user_item_matrix[:, j] > 0
                if mask.any():
                    centered_matrix[mask, j] = self.user_item_matrix[mask, j] - self.movie_means[j]
            
            # Fit Truncated SVD (optimal latent dimensions K = 20)
            n_comp = min(20, centered_matrix.shape[0] - 1, centered_matrix.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            user_factors = svd.fit_transform(centered_matrix)
            self.svd_components = svd.components_ # Shape: (K, M)
            
            # Reconstruct ratings matrix for database users
            self.reconstructed_matrix = (user_factors @ self.svd_components)
            # Re-add movie averages
            for j in range(self.reconstructed_matrix.shape[1]):
                self.reconstructed_matrix[:, j] += self.movie_means[j]
                
            logger.info(f"SVD model successfully decomposed matrix. Components shape: {self.svd_components.shape}")

    def save(self, filepath_prefix: str = "collab_model"):
        """
        Saves the fitted model attributes.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model must be fitted before saving.")
            
        try:
            model_data = {
                'kind': self.kind,
                'user_item_matrix': self.user_item_matrix,
                'similarity_matrix': self.similarity_matrix,
                'svd_components': self.svd_components,
                'reconstructed_matrix': self.reconstructed_matrix,
                'user_means': self.user_means,
                'movie_means': self.movie_means,
                'movie_id_to_col_idx': self.movie_id_to_col_idx,
                'col_idx_to_movie_id': self.col_idx_to_movie_id,
                'user_id_to_row_idx': self.user_id_to_row_idx,
                'row_idx_to_user_id': self.row_idx_to_user_id
            }
            path = config.MODELS_DIR / f"{filepath_prefix}_{self.kind}.joblib"
            joblib.dump(model_data, path)
            logger.info(f"Collaborative model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving collaborative model: {e}")

    def load(self, filepath_prefix: str = "collab_model") -> bool:
        """
        Loads the saved model files.
        """
        path = config.MODELS_DIR / f"{filepath_prefix}_{self.kind}.joblib"
        if not path.exists():
            logger.warning(f"Saved collaborative model not found at {path}")
            return False
            
        try:
            model_data = joblib.load(path)
            self.kind = model_data['kind']
            self.user_item_matrix = model_data['user_item_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.svd_components = model_data.get('svd_components')
            self.reconstructed_matrix = model_data.get('reconstructed_matrix')
            self.user_means = model_data['user_means']
            self.movie_means = model_data['movie_means']
            self.movie_id_to_col_idx = model_data['movie_id_to_col_idx']
            self.col_idx_to_movie_id = model_data['col_idx_to_movie_id']
            self.user_id_to_row_idx = model_data['user_id_to_row_idx']
            self.row_idx_to_user_id = model_data['row_idx_to_user_id']
            logger.info(f"Collaborative model ({self.kind}) loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading collaborative model: {e}")
            return False

    def predict_all_ratings_for_user(self, user_ratings_dict: dict[int, float]) -> np.ndarray:
        """
        Predicts ratings for all movies in the vocabulary for a target user profile.
        user_ratings_dict: dict of {movieId: rating} representing the user profile.
        Returns a numpy array of predicted ratings for all movies (length equal to columns/movies).
        """
        num_movies = len(self.movie_id_to_col_idx)
        predicted_ratings = np.zeros(num_movies)
        
        if not user_ratings_dict:
            # Fallback to movie mean ratings (popularity) if no user ratings are provided
            for i in range(num_movies):
                predicted_ratings[i] = self.movie_means[i] if i < len(self.movie_means) else 3.5
            return predicted_ratings

        # Convert user ratings dictionary to a vectorized user ratings array
        user_vector = np.zeros(num_movies)
        for m_id, rating in user_ratings_dict.items():
            if m_id in self.movie_id_to_col_idx:
                col_idx = self.movie_id_to_col_idx[m_id]
                user_vector[col_idx] = rating

        if self.kind == "item":
            # --- Vectorized Item-Based Prediction ---
            rated_mask = (user_vector > 0).astype(float)
            pos_sim_matrix = np.maximum(self.similarity_matrix, 0)
            numerator = pos_sim_matrix @ user_vector
            denominator = pos_sim_matrix @ rated_mask + 1e-9
            predicted_ratings = numerator / denominator
            
            for i in range(num_movies):
                if denominator[i] < 1e-5:
                    predicted_ratings[i] = self.movie_means[i]
                    
        elif self.kind == "user":
            # --- User-Based Prediction ---
            rated_indices = np.where(user_vector > 0)[0]
            if len(rated_indices) > 0:
                custom_user_mean = np.mean(user_vector[rated_indices])
                custom_centered = np.zeros(num_movies)
                custom_centered[rated_indices] = user_vector[rated_indices] - custom_user_mean
            else:
                custom_user_mean = 3.5
                custom_centered = np.zeros(num_movies)
                
            centered_db_matrix = np.zeros_like(self.user_item_matrix)
            for i in range(self.user_item_matrix.shape[0]):
                mask = self.user_item_matrix[i] > 0
                if mask.any():
                    centered_db_matrix[i, mask] = self.user_item_matrix[i, mask] - self.user_means[i]
            
            user_sims = cosine_similarity(custom_centered.reshape(1, -1), centered_db_matrix).flatten()
            k = min(40, len(user_sims))
            top_k_users = np.argsort(user_sims)[::-1][:k]
            
            for i in range(num_movies):
                sim_users_who_rated = [u for u in top_k_users if self.user_item_matrix[u, i] > 0 and user_sims[u] > 0]
                if not sim_users_who_rated:
                    predicted_ratings[i] = self.movie_means[i]
                else:
                    num = 0.0
                    den = 0.0
                    for u in sim_users_who_rated:
                        rating_centered = self.user_item_matrix[u, i] - self.user_means[u]
                        sim_weight = user_sims[u]
                        num += sim_weight * rating_centered
                        den += sim_weight
                    predicted_ratings[i] = custom_user_mean + (num / (den + 1e-9))
                    
        else:
            # --- TruncatedSVD Matrix Factorization Projection ---
            # Center user ratings before SVD projection
            centered_user_vector = np.zeros(num_movies)
            rated_indices = np.where(user_vector > 0)[0]
            if len(rated_indices) > 0:
                for idx in rated_indices:
                    centered_user_vector[idx] = user_vector[idx] - self.movie_means[idx]
            
            # Project vector into lower dimensional space: shape (1, K)
            latent_projection = centered_user_vector @ self.svd_components.T
            # Reconstruct back to feature space: shape (M,)
            reconstructed_user = latent_projection @ self.svd_components
            # Add movie averages back to predicted rating
            predicted_ratings = reconstructed_user + self.movie_means

        predicted_ratings = np.clip(predicted_ratings, 0.5, 5.0)
        return predicted_ratings

    def recommend_for_user(self, user_ratings_dict: dict[int, float], top_n: int = 5) -> list[tuple[int, float]]:
        """
        Recommends top_n movies for a user based on collaborative filtering predicted ratings.
        """
        if self.user_item_matrix is None:
            raise ValueError("Model must be fitted before saving.")
            
        predicted_ratings = self.predict_all_ratings_for_user(user_ratings_dict)
        sorted_indices = np.argsort(predicted_ratings)[::-1]
        rated_movie_ids = set(user_ratings_dict.keys())
        
        recommendations = []
        for idx in sorted_indices:
            movie_id = self.col_idx_to_movie_id[idx]
            if movie_id not in rated_movie_ids:
                recommendations.append((movie_id, float(predicted_ratings[idx])))
            if len(recommendations) == top_n:
                break
                
        return recommendations
