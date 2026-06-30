# Cinematique: Movie Recommendation System

An AI-powered Movie Recommendation System that utilizes hybrid machine learning techniques (Content-Based and Collaborative Filtering), dynamically adjustments using review sentiment analysis, SQLite storage, and a modern Streamlit user interface featuring interactive visual analytics.

This project is structured as a professional-grade software repository, suitable for internships, developer portfolios, and technical interviews.

---

## Key Features

- **Multi-Algorithm Recommendation Engine**:
  - **Content-Based Filtering**: Recommends movies matching genres and tag overlays using TF-IDF Vectorization and Cosine Similarity.
  - **Collaborative Filtering**: Recommends movies matching similar user rating histories, supporting both **Item-Based** and **User-Based** Nearest Neighbors algorithms.
  - **Hybrid Blending**: Integrates content similarity and collaborative prediction scores using a normalized weighted averages calculation.
- **Dynamic Sentiment-Based Boosts**: 
  - Submits textual movie reviews directly in the UI.
  - Classifies review sentiments as *Positive*, *Neutral*, or *Negative* using a rule-based lexicon processor.
  - Adjusts recommendation ranks by boosting or penalizing scores according to aggregate viewer review sentiments.
- **Interactive Analytics Dashboard**:
  - Displays KPI counts (Total users, ratings, movies, watchlist).
  - Renders interactive Plotly visual charts for genre distribution, rating distribution, movie release history, and a Jaccard-style genre co-occurrence correlation heatmap.
- **Persistent SQLite Database**:
  - Saves custom user ratings, watchlist bookmarks, search query history, and recommendations logs.
  - Dynamically merges local user ratings at query-time into the collaborative calculations for real-time recommendation updates.
- **Modular Production Architecture**:
  - Clean separation of data pipelines, mathematical models, SQLite helpers, visualizations, and UI layouts.
  - Unit tests covering title extraction, sentiment analysis, similarity calculations, database, and hybrid pipelines.

---

## Directory Structure

```text
movie-recommendation-system/
│
├── app.py                     # Entrypoint script (programmatic streamlit loader)
├── streamlit_app.py           # Main Streamlit web application
├── config.py                  # Directory paths and model hyperparameters settings
├── requirements.txt           # Package dependencies list
├── .gitignore                 # Files excluded from Git tracking
├── LICENSE                    # MIT License open-source file
│
├── data/                      # Contains downloaded dataset CSV files & SQLite DB
│   └── movie_recs.db          # SQLite local data store
│
├── models/                    # Recommendation model modules
│   ├── content_filter.py      # TF-IDF & Cosine Similarity logic
│   ├── collaborative_filter.py# Item-Based & User-Based ratings logic
│   ├── hybrid_model.py        # Blending logic with explanation generation
│   └── saved/                 # Serialized model checkpoints
│
├── utils/                     # Utility packages
│   ├── preprocessing.py       # Auto-downloads, extracts, and cleans MovieLens data
│   ├── helpers.py             # SQLite query operations
│   ├── recommendation.py      # High-level Router and Lexicon Sentiment Analyzer
│   └── visualization.py       # Plotly analytics plotting
│
├── notebook/                  # Jupyter notebooks for development
│   └── Movie_Recommendation_EDA.ipynb
│
├── reports/                   # Technical submission writeups
│   └── internship_report.md   # Project submission report
│
└── tests/                     # Automated testing folder
    └── test_recommendation.py # pytest unit tests
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- `pip` (Python package manager)

### Setup Instructions

1. **Clone or navigate into the project directory**:
   ```bash
   cd c:/Users/Meena/MOVIE_RECOMMENDATION_SYSTEM
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   You can start the Streamlit application programmatically using the python script:
   ```bash
   python app.py
   ```
   Or execute standard streamlit runner:
   ```bash
   streamlit run streamlit_app.py
   ```
   *Note: On first startup, the application will automatically download the 1MB MovieLens small dataset zip file, extract `movies.csv`, `ratings.csv`, and `tags.csv` directly into the `data/` directory, and fit the similarity models. No manual downloads are necessary.*

---

## Running Automated Unit Tests

Automated testing is implemented using `pytest` to verify the mathematical calculations, database read/writes, and text-based sentiment negations. Run tests with:
```bash
pytest tests/test_recommendation.py
```
Or run as python module:
```bash
python -m pytest tests/test_recommendation.py
```

---
