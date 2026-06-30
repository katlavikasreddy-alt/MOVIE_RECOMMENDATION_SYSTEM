import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# Apply general clean dark styling theme to all plots
PLOT_TEMPLATE = "plotly_dark"
COLOR_PALETTE = px.colors.sequential.Plasma_r

def plot_genre_distribution(movies_df: pd.DataFrame) -> go.Figure:
    """
    Plots the distribution of movies across different genres.
    """
    # Split genres and explode to get individual genre counts
    genres_series = movies_df['genres'].str.split('|')
    all_genres = genres_series.explode()
    genre_counts = all_genres.value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    # Filter out empty or placeholder genres
    genre_counts = genre_counts[genre_counts['Genre'] != '(no genres listed)']
    
    fig = px.bar(
        genre_counts,
        x='Count',
        y='Genre',
        orientation='h',
        title="Number of Movies by Genre",
        color='Count',
        color_continuous_scale=px.colors.sequential.Sunsetdark,
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    return fig

def plot_rating_distribution(ratings_df: pd.DataFrame) -> go.Figure:
    """
    Plots a histogram showing the distribution of movie ratings.
    """
    fig = px.histogram(
        ratings_df,
        x='rating',
        nbins=10,
        title="Distribution of Ratings",
        color_discrete_sequence=['#ff4b4b'],
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        xaxis_title="Rating Value",
        yaxis_title="Count",
        bargap=0.1,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_top_rated_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, min_ratings: int = 50, top_n: int = 10) -> go.Figure:
    """
    Plots the top_n highest-rated movies with at least min_ratings.
    """
    # Compute mean rating and count per movie
    stats = ratings_df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Filter by minimum ratings
    filtered_stats = stats[stats['rating_count'] >= min_ratings]
    
    # Merge with movies to get titles
    merged = pd.merge(filtered_stats, movies_df, on='movieId')
    top_movies = merged.sort_values(by='avg_rating', ascending=False).head(top_n)
    
    fig = px.bar(
        top_movies,
        x='avg_rating',
        y='title',
        orientation='h',
        title=f"Top {top_n} Highest Rated Movies (Min {min_ratings} Ratings)",
        color='avg_rating',
        color_continuous_scale=px.colors.sequential.Electric,
        labels={'avg_rating': 'Avg Rating', 'title': 'Movie Title'},
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis=dict(range=[1, 5]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

def plot_popular_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Plots the top_n most-rated (popular) movies.
    """
    stats = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    merged = pd.merge(stats, movies_df, on='movieId')
    popular_movies = merged.sort_values(by='rating_count', ascending=False).head(top_n)
    
    fig = px.bar(
        popular_movies,
        x='rating_count',
        y='title',
        orientation='h',
        title=f"Top {top_n} Most Popular Movies (By Rating Count)",
        color='rating_count',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'rating_count': 'Rating Count', 'title': 'Movie Title'},
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

def plot_genre_cooccurrence(movies_df: pd.DataFrame) -> go.Figure:
    """
    Plots a heatmap showing co-occurrence correlation of movie genres.
    """
    # Explode genres
    df = movies_df.copy()
    df['genre_list'] = df['genres'].str.split('|')
    
    # Get set of all unique genres
    unique_genres = sorted(list(set(g for genres in df['genre_list'] for g in genres if g != '(no genres listed)')))
    
    # Initialize co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=unique_genres, columns=unique_genres)
    
    # Fill matrix
    for genres in df['genre_list']:
        active = [g for g in genres if g in co_matrix.index]
        for g1 in active:
            for g2 in active:
                co_matrix.loc[g1, g2] += 1
                
    # Normalize by converting to correlation-like ratio (Jaccard similarity style)
    # intersection(A, B) / union(A, B)
    norm_matrix = pd.DataFrame(0.0, index=unique_genres, columns=unique_genres)
    for g1 in unique_genres:
        for g2 in unique_genres:
            if g1 == g2:
                norm_matrix.loc[g1, g2] = 1.0
            else:
                intersection = co_matrix.loc[g1, g2]
                union = co_matrix.loc[g1, g1] + co_matrix.loc[g2, g2] - intersection
                norm_matrix.loc[g1, g2] = intersection / union if union > 0 else 0.0

    fig = px.imshow(
        norm_matrix,
        labels=dict(x="Genre", y="Genre", color="Overlap Ratio"),
        x=unique_genres,
        y=unique_genres,
        title="Genre Co-Occurrence Heatmap (Overlap Ratio)",
        color_continuous_scale=px.colors.sequential.Magma,
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    return fig

def plot_movies_by_year(movies_df: pd.DataFrame) -> go.Figure:
    """
    Plots movie releases count trend over the years.
    """
    # Filter out year = 0
    df = movies_df[movies_df['year'] > 1900].copy()
    year_counts = df.groupby('year').size().reset_index(name='count')
    
    fig = px.area(
        year_counts,
        x='year',
        y='count',
        title="Movie Releases Over Time (by Year)",
        color_discrete_sequence=['#00cc96'],
        template=PLOT_TEMPLATE
    )
    fig.update_layout(
        xaxis_title="Release Year",
        yaxis_title="Number of Movies",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
