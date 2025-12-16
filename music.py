import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

# Page config
st.set_page_config(page_title="KenyaBeats Music Recommender", layout="wide")

# Load dataset (Spotify‑Africa master tracks)
@st.cache_data
def load_data():
    dataset = load_dataset("electricsheepafrica/master_tracks")
    df = dataset["train"].to_pandas()
    return df

df = load_data()

# Filter for Kenyan tracks
df_kenya = df[df["country"] == "KE"]  # Kenya country code in dataset :contentReference[oaicite:1]{index=1}

# Header with gradient
st.markdown("""
    <div style="background: linear-gradient(90deg, #ff7e5f, #feb47b); padding: 1.5rem 0; border-radius: .75rem;">
        <h1 style="text-align: center; color: white; margin: 0;">Discover Kenyan Music</h1>
    </div>
""", unsafe_allow_html=True)

# Search bar
search_input = st.text_input("Search for artist or track", "")

# Filter search
if search_input:
    df_search = df_kenya[
        df_kenya["track_name"].str.contains(search_input, case=False, na=False) |
        df_kenya["artist_name"].str.contains(search_input, case=False, na=False)
    ]
else:
    df_search = df_kenya.copy()

# Stats cards
tracks_count = len(df_search)
artists_count = df_search["artist_name"].nunique()
avg_popularity = round(df_search["popularity"].mean(), 2) if tracks_count else 0
recs_count = 0  # We'll fill this after recommendation logic

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracks", tracks_count)
col2.metric("Artists", artists_count)
col3.metric("Avg. Popularity", avg_popularity)
col4.metric("Recs (placeholder)", recs_count)

# Tabs: Tracks / Artists / Analytics
tab1, tab2, tab3 = st.tabs(["Tracks", "Artists", "Analytics"])

with tab1:
    st.write("### Tracks")
    st.dataframe(df_search[["track_name", "artist_name", "popularity"]].reset_index(drop=True))

with tab2:
    st.write("### Artists")
    st.dataframe(df_search[["artist_name"]].drop_duplicates().reset_index(drop=True))

with tab3:
    st.write("### Audio Feature Distributions")

    # Pick some audio feature columns if present
    features = ["popularity", "duration_ms"]
    available = [f for f in features if f in df_search.columns]

    for feat in available:
        fig = px.histogram(df_search, x=feat, title=f"{feat} distribution")
        st.plotly_chart(fig, use_container_width=True)

# Simple content‑based recommendations
if tracks_count >= 3 and all(f in df_search.columns for f in ["popularity", "duration_ms"]):
    feat_cols = ["popularity", "duration_ms"]
    X = df_search[feat_cols].fillna(0)
    model = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(X)
    user_idx = list(df_search.index[:1])

    distances, indices = model.kneighbors(X.loc[user_idx])
    recs = df_search.iloc[indices[0]]
    recs_count = len(recs)

    st.write("## Recommended For You")
    st.dataframe(recs[["track_name", "artist_name", "popularity"]].reset_index(drop=True))

    # Update recs metric
    col4.metric("Recs (approx)", recs_count)

