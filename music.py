import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

# For social media scraping
import snscrape.modules.twitter as sntwitter
from youtubesearchpython import VideosSearch

# ------------------------------
# Load Spotify-Africa Dataset
# ------------------------------
@st.cache_data
def load_data():
    dataset = load_dataset("electricsheepafrica/master_tracks")
    df = dataset["train"].to_pandas()
    return df

df = load_data()
df_kenya = df[df["country"] == "KE"].copy()

# ------------------------------
# Streamlit Header
# ------------------------------
st.markdown("""
    <div style="background: linear-gradient(90deg, #ff7e5f, #feb47b); padding: 1.5rem 0; border-radius: .75rem;">
        <h1 style="text-align: center; color: white; margin: 0;">Discover Kenyan Music</h1>
    </div>
""", unsafe_allow_html=True)

# ------------------------------
# Search bar
# ------------------------------
search_input = st.text_input("Search for artist or track", "")

if search_input:
    df_search = df_kenya[
        df_kenya["track_name"].str.contains(search_input, case=False, na=False) |
        df_kenya["artist_name"].str.contains(search_input, case=False, na=False)
    ]
else:
    df_search = df_kenya.copy()

# ------------------------------
# Social Media Scraping Functions
# ------------------------------
@st.cache_data(show_spinner=False)
def scrape_twitter_score(artist_name, max_tweets=30):
    """Scrape tweets and calculate a social score."""
    score = 0
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'"{artist_name}" lang:en').get_items()):
            if i >= max_tweets:
                break
            score += tweet.likeCount + tweet.retweetCount
    except Exception as e:
        st.warning(f"Twitter scraping error for {artist_name}: {e}")
    return score

@st.cache_data(show_spinner=False)
def scrape_youtube_score(artist_name, max_videos=5):
    """Scrape YouTube trending music videos for an artist."""
    score = 0
    try:
        videosSearch = VideosSearch(artist_name, limit=max_videos)
        results = videosSearch.result().get("result", [])
        for video in results:
            views = int(video.get("viewCount", {}).get("text", "0").replace(",", "").split()[0])
            score += views
    except Exception as e:
        st.warning(f"YouTube scraping error for {artist_name}: {e}")
    return score

# ------------------------------
# Compute social score per artist
# ------------------------------
with st.spinner("Scraping social media for trending scores..."):
    social_scores = {}
    for artist in df_search["artist_name"].unique():
        twitter_score = scrape_twitter_score(artist)
        youtube_score = scrape_youtube_score(artist)
        social_scores[artist] = twitter_score + youtube_score

df_search["social_score"] = df_search["artist_name"].map(social_scores).fillna(0)

# ------------------------------
# Stats Cards
# ------------------------------
tracks_count = len(df_search)
artists_count = df_search["artist_name"].nunique()
avg_popularity = round(df_search["popularity"].mean(), 2)
avg_social = int(df_search["social_score"].mean())
recs_count = 0  # will update after recommendations

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracks", tracks_count)
col2.metric("Artists", artists_count)
col3.metric("Avg. Popularity", avg_popularity)
col4.metric("Avg. Social Score", avg_social)

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["Tracks", "Artists", "Analytics"])

with tab1:
    st.write("### Tracks")
    st.dataframe(df_search[["track_name", "artist_name", "popularity", "social_score"]].reset_index(drop=True))

with tab2:
    st.write("### Artists")
    st.dataframe(df_search[["artist_name", "social_score"]].drop_duplicates().reset_index(drop=True))

with tab3:
    st.write("### Audio Feature Distributions")
    features = ["popularity", "duration_ms", "social_score"]
    for feat in features:
        if feat in df_search.columns:
            fig = px.histogram(df_search, x=feat, title=f"{feat} distribution")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Hybrid Recommendation (Audio + Social Score)
# ------------------------------
if tracks_count >= 3:
    feature_cols = ["popularity", "duration_ms", "social_score"]
    X = df_search[feature_cols].fillna(0)
    model = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(X)
    user_idx = list(df_search.index[:1])  # first track as example

    distances, indices = model.kneighbors(X.loc[user_idx])
    recs = df_search.iloc[indices[0]]
    recs_count = len(recs)

    st.write("## Recommended For You (Trending + Audio Features)")
    st.dataframe(recs[["track_name", "artist_name", "popularity", "social_score"]].reset_index(drop=True))

    col4.metric("Recs Count", recs_count)
