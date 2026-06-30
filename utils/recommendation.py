import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import get_average_sentiment_boost, add_recommendation_history
from models.hybrid_model import HybridRecommender

logger = logging.getLogger(__name__)

# Simple, robust sentiment dictionary for rule-based sentiment analysis
POSITIVE_WORDS = {
    'great', 'excellent', 'love', 'loved', 'amazing', 'wonderful', 'masterpiece',
    'fantastic', 'good', 'best', 'brilliant', 'classic', 'beautiful', 'must-watch',
    'fun', 'enjoyed', 'enjoyable', 'awesome', 'masterful', 'perfect', 'superb',
    'hilarious', 'entertaining', 'gripping', 'charming', 'clever', 'spectacular'
}

NEGATIVE_WORDS = {
    'bad', 'worst', 'hate', 'hated', 'terrible', 'awful', 'boring', 'waste',
    'poor', 'disappointed', 'disappointing', 'garbage', 'trash', 'slow',
    'stupid', 'predictable', 'overrated', 'annoying', 'dumb', 'lame',
    'unwatchable', 'pointless', 'weak', 'horrible', 'flat', 'cliche'
}

NEGATIONS = {'not', 'no', 'never', 'didnt', 'wasnt', 'cant', 'couldnt', 'isnt', 'arent', 'neither', 'nor'}

def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Performs rule-based sentiment analysis on textual reviews.
    Supports basic negation handling (e.g., 'not good' -> negative, 'not bad' -> positive).
    Returns (sentiment_label, sentiment_score)
    where score is in [-1.0, 1.0].
    """
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0
        
    # Lowercase and split words, removing basic punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    if not tokens:
        return "Neutral", 0.0
        
    pos_score = 0
    neg_score = 0
    
    for i, token in enumerate(tokens):
        is_negated = False
        # Check if preceding token (or token before that, separated by common modifiers) is a negation
        if i > 0 and tokens[i-1] in NEGATIONS:
            is_negated = True
        elif i > 1 and tokens[i-2] in NEGATIONS and tokens[i-1] in {'a', 'an', 'the', 'very', 'extremely', 'really', 'too', 'particularly', 'so'}:
            is_negated = True
            
        if token in POSITIVE_WORDS:
            if is_negated:
                neg_score += 1
            else:
                pos_score += 1
        elif token in NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1
            else:
                neg_score += 1
                
    total = pos_score + neg_score
    if total == 0:
        return "Neutral", 0.0
        
    score = (pos_score - neg_score) / float(total)
    
    # Classify label based on score threshold
    if score > 0.15:
        label = "Positive"
    elif score < -0.15:
        label = "Negative"
    else:
        label = "Neutral"
        
    return label, float(score)


class RecommendationManager:
    def __init__(self, collab_kind: str = "item"):
        self.hybrid_engine = HybridRecommender(collab_kind=collab_kind)
        self.is_fitted = False

    def train_models(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Fits the underlying recommendation models.
        """
        self.hybrid_engine.fit(movies_df, ratings_df)
        self.is_fitted = True

    def get_recommendations(
        self,
        user_ratings_dict: dict[int, float],
        method: str = "hybrid",
        w_content: float = 0.5,
        w_collab: float = 0.5,
        enable_sentiment: bool = True,
        top_n: int = 5
    ) -> list[dict]:
        """
        High-level recommendations router: Content, Collaborative, Hybrid,
        with optional Sentiment-Based Ranking Adjustments.
        """
        if not self.is_fitted:
            raise ValueError("Recommendation models are not trained yet.")
            
        # Determine weights based on method
        if method == "content":
            weight_content, weight_collab = 1.0, 0.0
        elif method == "collab":
            weight_content, weight_collab = 0.0, 1.0
        else: # hybrid
            weight_content, weight_collab = w_content, w_collab

        # Retrieve baseline hybrid model recommendations (fetch 2-3x top_n for sentiment re-ranking)
        fetch_n = top_n * 3 if enable_sentiment else top_n
        raw_recs = self.hybrid_engine.recommend_hybrid(
            user_ratings_dict, 
            w_content=weight_content, 
            w_collab=weight_collab, 
            top_n=fetch_n
        )
        
        # Apply sentiment adjustment if enabled
        adjusted_recs = []
        for rec in raw_recs:
            movie_id = rec['movieId']
            # Get average review sentiment score from database
            avg_sentiment = get_average_sentiment_boost(movie_id)
            
            base_score = rec['hybrid_score']
            base_rating = rec['predicted_rating']
            
            if enable_sentiment and abs(avg_sentiment) > 0.01:
                # Adjust hybrid score
                sentiment_effect = avg_sentiment * config.SENTIMENT_BOOST_MULTIPLIER
                adjusted_score = np.clip(base_score + sentiment_effect, 0.0, 1.0)
                adjusted_rating = np.clip(base_rating + (sentiment_effect * 4.5), 0.5, 5.0)
                
                # Update reason and scores
                if avg_sentiment > 0.15:
                    reason = rec['reason'] + " Boosted by positive review sentiments."
                elif avg_sentiment < -0.15:
                    reason = rec['reason'] + " Penalized due to negative review sentiments."
                else:
                    reason = rec['reason']
                    
                rec['hybrid_score'] = float(adjusted_score)
                rec['predicted_rating'] = float(adjusted_rating)
                rec['reason'] = reason
                rec['sentiment_score'] = avg_sentiment
            else:
                rec['sentiment_score'] = 0.0
                
            adjusted_recs.append(rec)
            
        # Re-sort if sentiment adjustment took place
        if enable_sentiment:
            adjusted_recs = sorted(adjusted_recs, key=lambda x: x['hybrid_score'], reverse=True)
            
        # Slice to user's desired top_n
        final_recs = adjusted_recs[:top_n]
        
        # Re-assign rank order
        for idx, rec in enumerate(final_recs):
            rec['rank'] = idx + 1
            
        # Log recommendations history in SQLite database (for USER_LOCAL_ID)
        rec_ids = [rec['movieId'] for rec in final_recs]
        if rec_ids:
            add_recommendation_history(config.USER_LOCAL_ID, method, rec_ids)
            
        return final_recs
