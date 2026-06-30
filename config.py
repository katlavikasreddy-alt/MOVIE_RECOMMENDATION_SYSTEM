import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
DB_PATH = DATA_DIR / "movie_recs.db"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset URLs & Configuration
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_FILE_NAME = "ml-latest-small.zip"
EXTRACTED_FOLDER_NAME = "ml-latest-small"

# Model Hyperparameters
DEFAULT_NUM_RECOMMENDATIONS = 5
HYBRID_CONTENT_WEIGHT = 0.5
HYBRID_COLLAB_WEIGHT = 0.5

# Sentiment-based boost settings
SENTIMENT_BOOST_MULTIPLIER = 0.15 # Max adjustment to predicted rating

# SQLite database setup configurations
USER_LOCAL_ID = 99999  # Special User ID reserved for local UI user in ratings calculations
