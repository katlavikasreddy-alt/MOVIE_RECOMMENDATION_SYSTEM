import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.content_filter import ContentRecommender
from models.collaborative_filter import CollaborativeRecommender

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, collab_kind: str = "item"):
        self.content_model = ContentRecommender()
        self.collab_model = CollaborativeRecommender(kind=collab_kind)
        self.movies_df = None
        self.ratings_count = {}
        self.movie_avg_ratings = {}

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Fits both Content-Based and Collaborative Filtering models.
        """
        logger.info("Fitting Hybrid Recommendation Engine...")
        self.movies_df = movies_df.copy()
        
        # Fit underlying models
        self.content_model.fit(self.movies_df)
        self.collab_model.fit(ratings_df)
        
        # Precompute popularity metrics for recommendations output
        # Popularity score is represented by number of ratings a movie has
        counts = ratings_df.groupby('movieId').size().to_dict()
        self.ratings_count = {m_id: counts.get(m_id, 0) for m_id in self.movies_df['movieId']}
        
        averages = ratings_df.groupby('movieId')['rating'].mean().to_dict()
        self.movie_avg_ratings = {m_id: averages.get(m_id, 0.0) for m_id in self.movies_df['movieId']}
        
        logger.info("Hybrid Recommendation Engine fitted successfully.")

    def save(self, content_prefix: str = "content_model", collab_prefix: str = "collab_model"):
        """
        Saves both Content and Collaborative model checkpoints.
        """
        self.content_model.save(content_prefix)
        self.collab_model.save(collab_prefix)

    def load(self, content_prefix: str = "content_model", collab_prefix: str = "collab_model") -> bool:
        """
        Loads both Content and Collaborative model checkpoints.
        """
        c_success = self.content_model.load(content_prefix)
        col_success = self.collab_model.load(collab_prefix)
        
        if c_success and col_success:
            self.movies_df = self.content_model.movies_df
            # Compute counts and averages from the collaborative ratings matrix if needed, 
            # but usually movies_df is loaded. Let's make sure movies_df is present.
            # We'll compute ratings_count and movie_avg_ratings from collaborative model attributes.
            col_movies = list(self.collab_model.movie_id_to_col_idx.keys())
            self.ratings_count = {m_id: int(np.sum(self.collab_model.user_item_matrix[:, self.collab_model.movie_id_to_col_idx[m_id]] > 0)) 
                                  if m_id in self.collab_model.movie_id_to_col_idx else 0 for m_id in self.movies_df['movieId']}
            self.movie_avg_ratings = {m_id: float(self.collab_model.movie_means[self.collab_model.movie_id_to_col_idx[m_id]])
                                      if m_id in self.collab_model.movie_id_to_col_idx else 0.0 for m_id in self.movies_df['movieId']}
            return True
        return False

    def recommend_hybrid(
        self, 
        user_ratings_dict: dict[int, float], 
        w_content: float = 0.5, 
        w_collab: float = 0.5, 
        top_n: int = 5
    ) -> list[dict]:
        """
        Generates hybrid recommendations by taking weighted scores from Content and Collaborative Filtering.
        """
        if self.movies_df is None:
            raise ValueError("Hybrid engine must be fitted before recommendations can be made.")
            
        num_movies = len(self.movies_df)
        
        # 1. Get Collaborative Filtering predicted ratings for all movies
        # Array shape: (M,)
        collab_predictions = self.collab_model.predict_all_ratings_for_user(user_ratings_dict)
        
        # 2. Get Content Filtering scores for all movies
        # Compute user content vector
        content_scores = np.zeros(num_movies)
        if user_ratings_dict and self.content_model.tfidf_matrix is not None:
            profile_vector = np.zeros(self.content_model.tfidf_matrix.shape[1])
            weight_sum = 0.0
            
            for m_id, r in user_ratings_dict.items():
                if m_id in self.content_model.movie_id_to_idx:
                    idx = self.content_model.movie_id_to_idx[m_id]
                    movie_vector = self.content_model.tfidf_matrix[idx].toarray().flatten()
                    weight = r - 2.5 # Weight positive ratings positively and negative negatively
                    profile_vector += movie_vector * weight
                    weight_sum += abs(weight)
                    
            if weight_sum > 0 and not np.all(profile_vector == 0):
                profile_vector = profile_vector / np.linalg.norm(profile_vector)
                # Compute similarities for all movies
                from sklearn.metrics.pairwise import linear_kernel
                content_scores = linear_kernel(profile_vector.reshape(1, -1), self.content_model.tfidf_matrix).flatten()
                
        # 3. Normalization of scores
        # Collaborative ratings are normally [0.5, 5.0]. Min-Max normalize them:
        collab_min = collab_predictions.min()
        collab_max = collab_predictions.max()
        collab_range = collab_max - collab_min if collab_max > collab_min else 1.0
        collab_norm = (collab_predictions - collab_min) / collab_range
        
        # Content scores are already [0, 1] or similar, let's min-max normalize them to [0, 1] as well
        content_min = content_scores.min()
        content_max = content_scores.max()
        content_range = content_max - content_min if content_max > content_min else 1.0
        content_norm = (content_scores - content_min) / content_range
        
        # 4. Blend scores
        hybrid_scores = (w_content * content_norm) + (w_collab * collab_norm)
        
        # 5. Sort recommendations (excluding movies user has rated)
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        rated_movie_ids = set(user_ratings_dict.keys())
        
        recommendations = []
        rank = 1
        
        for idx in sorted_indices:
            movie_row = self.movies_df.iloc[idx]
            movie_id = movie_row['movieId']
            
            # Skip already rated movies
            if movie_id in rated_movie_ids:
                continue
                
            c_score = float(content_scores[idx])
            p_rating = float(collab_predictions[self.collab_model.movie_id_to_col_idx[movie_id]]) if movie_id in self.collab_model.movie_id_to_col_idx else 0.0
            h_score = float(hybrid_scores[idx])
            
            # Generate explainable recommendation reason
            genres_list = movie_row['genres'].split('|')
            primary_genres = ", ".join(genres_list[:2])
            
            # Determine logic for explanation
            if w_content > 0.7:
                reason = f"Matches your genre interests ({primary_genres}) and favorite tags."
            elif w_collab > 0.7:
                reason = "Highly recommended by users with similar movie tastes."
            else:
                if c_score > 0.4 and p_rating >= 4.0:
                    reason = f"Perfect blend: matches your favorite themes ({primary_genres}) and is highly rated by similar users."
                elif c_score > 0.4:
                    reason = f"Strong match for your preferred movie features in {primary_genres}."
                else:
                    reason = "Recommended based on positive ratings from users with similar viewing patterns."

            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_row['title'],
                'clean_title': movie_row['clean_title'],
                'genres': movie_row['genres'],
                'year': int(movie_row['year']),
                'similarity_score': c_score,
                'predicted_rating': p_rating,
                'hybrid_score': h_score,
                'rank': rank,
                'popularity_score': self.ratings_count.get(movie_id, 0),
                'average_rating': self.movie_avg_ratings.get(movie_id, 0.0),
                'reason': reason
            })
            
            rank += 1
            if len(recommendations) == top_n:
                break
                
        return recommendations
