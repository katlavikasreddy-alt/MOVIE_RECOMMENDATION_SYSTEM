import os
import zipfile
import urllib.request
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add base directory to path so config can be imported when running scripts directly
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_DIR / "preprocessing.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def download_and_extract_dataset() -> bool:
    """
    Downloads and extracts the MovieLens small dataset if files do not exist.
    Returns True if successfully downloaded/extracted or already present, False otherwise.
    """
    required_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    all_exist = all((config.DATA_DIR / f).exists() for f in required_files)
    
    if all_exist:
        logger.info("Dataset files already exist in data directory. Skipping download.")
        return True

    zip_path = config.DATA_DIR / config.ZIP_FILE_NAME
    logger.info(f"Downloading dataset from {config.MOVIELENS_URL}...")
    try:
        urllib.request.urlretrieve(config.MOVIELENS_URL, zip_path)
        logger.info("Download completed successfully.")
        
        logger.info("Extracting dataset files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
        
        # Move files from extracted folder to data root
        extracted_dir = config.DATA_DIR / config.EXTRACTED_FOLDER_NAME
        for filename in required_files:
            source = extracted_dir / filename
            dest = config.DATA_DIR / filename
            if source.exists():
                if dest.exists():
                    os.remove(dest)
                os.rename(source, dest)
                logger.info(f"Moved {filename} to data directory.")

        # Cleanup zip and extracted subfolder
        if zip_path.exists():
            os.remove(zip_path)
        if extracted_dir.exists():
            import shutil
            shutil.rmtree(extracted_dir)
            
        logger.info("Dataset extracted and cleaned up successfully.")
        return True
    except Exception as e:
        logger.error(f"Error downloading or extracting dataset: {e}", exc_info=True)
        return False

def clean_title(title: str) -> str:
    """
    Cleans movie title by removing the year, brackets, and extra spaces.
    Example: "Toy Story (1995)" -> "Toy Story"
    """
    if not isinstance(title, str):
        return ""
    # Strip (Year) e.g., (1995)
    cleaned = re.sub(r'\s*\(\d{4}\)', '', title)
    return cleaned.strip()

def extract_year(title: str) -> int:
    """
    Extracts the release year from the movie title.
    Example: "Toy Story (1995)" -> 1995. Returns 0 if no year found.
    """
    if not isinstance(title, str):
        return 0
    match = re.search(r'\((\d{4})\)', title)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0

def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the movies dataframe: extracts year, cleans titles, replaces genres pipe character.
    """
    df = movies_df.copy()
    
    # Handle missing values
    df['title'] = df['title'].fillna("Unknown Movie")
    df['genres'] = df['genres'].fillna("(no genres listed)")
    
    # Extract year
    df['year'] = df['title'].apply(extract_year)
    
    # Clean title
    df['clean_title'] = df['title'].apply(clean_title)
    
    # Format genres for TF-IDF representation (space separated instead of pipes)
    # E.g. "Adventure|Animation|Children" -> "Adventure Animation Children"
    df['genres_processed'] = df['genres'].str.replace('|', ' ', regex=False)
    df['genres_processed'] = df['genres_processed'].replace('(no genres listed)', '')
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['movieId'])
    
    logger.info(f"Preprocessed {len(df)} movies.")
    return df

def preprocess_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans tag strings.
    """
    df = tags_df.copy()
    df['tag'] = df['tag'].fillna("").astype(str).str.strip().str.lower()
    df = df[df['tag'] != ""]
    return df

def get_merged_metadata(movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines movies with aggregated user tags to form metadata strings for Content Filtering.
    """
    # Group tags by movieId and join as space-separated text
    grouped_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(set(x))).reset_index()
    grouped_tags.rename(columns={'tag': 'user_tags'}, inplace=True)
    
    # Merge with movies
    df = pd.merge(movies_df, grouped_tags, on='movieId', how='left')
    df['user_tags'] = df['user_tags'].fillna('')
    
    # Combine genres and tags into a single metadata column
    df['metadata'] = df['genres_processed'] + " " + df['user_tags']
    # Clean extra whitespaces
    df['metadata'] = df['metadata'].str.strip().str.replace(r'\s+', ' ', regex=True)
    
    # Handle movies that end up with empty metadata
    df.loc[df['metadata'] == "", 'metadata'] = "movie"
    
    return df

def load_and_preprocess_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the entire loading and preprocessing pipeline.
    Returns (movies_df, ratings_df, tags_df) preprocessed dataframes.
    """
    # 1. Download and extract
    success = download_and_extract_dataset()
    if not success:
        raise FileNotFoundError("Could not locate or download the required MovieLens dataset files.")
    
    # 2. Read CSVs
    try:
        movies_raw = pd.read_csv(config.DATA_DIR / "movies.csv")
        ratings_raw = pd.read_csv(config.DATA_DIR / "ratings.csv")
        tags_raw = pd.read_csv(config.DATA_DIR / "tags.csv")
        logger.info("Loaded movies, ratings, and tags CSV files successfully.")
    except Exception as e:
        logger.error(f"Error loading CSV files: {e}")
        raise
        
    # 3. Preprocess
    movies_df = preprocess_movies(movies_raw)
    tags_df = preprocess_tags(tags_raw)
    
    # Ratings clean
    ratings_df = ratings_raw.drop_duplicates(subset=['userId', 'movieId']).copy()
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    
    # 4. Create metadata merged movies
    movies_metadata_df = get_merged_metadata(movies_df, tags_df)
    
    logger.info("Completed full preprocessing pipeline.")
    return movies_metadata_df, ratings_df, tags_df

if __name__ == "__main__":
    # Test script execution
    movies, ratings, tags = load_and_preprocess_all()
    print("Movies Sample:")
    print(movies[['title', 'clean_title', 'year', 'genres_processed', 'metadata']].head())
    print(f"Total Movies: {len(movies)}, Ratings: {len(ratings)}, Tags: {len(tags)}")
