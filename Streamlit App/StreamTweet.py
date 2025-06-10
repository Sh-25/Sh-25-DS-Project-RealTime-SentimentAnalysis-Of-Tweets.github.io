import pandas as pd
import re
import langdetect
from textblob import TextBlob
import streamlit as st
import plotly.express as px
from langdetect import detect

# Page setup
st.set_page_config(page_title="Real-time Tweet Sentiment Analysis", layout="wide")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned-RealTime-Tweet.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    return df.dropna(subset=['Timestamp'])
df = load_data()

# Sentiment classification
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Tweet_text'].apply(get_sentiment)

# Sidebar filters
st.sidebar.title("🔎 Filters")
selected_sentiments = st.sidebar.multiselect("Sentiment", df['Sentiment'].unique(), default=list(df['Sentiment'].unique()))
selected_languages = st.sidebar.multiselect("Language", df['Language'].unique(), default=list(df['Language'].unique()))

# Filtered DataFrame
filtered_df = df[(df['Sentiment'].isin(selected_sentiments)) & (df['Language'].isin(selected_languages))]

# Title
st.title("📊 Real-time Sentiment Analysis of Tweets")
st.markdown("Explore multilingual sentiment across **Kannada**, **Hindi**, and **English** tweets.")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("🗂️ Total Tweets", len(filtered_df))
col2.metric("🌐 Languages", filtered_df['Language'].nunique())
col3.metric("📍 Locations", filtered_df['Location'].nunique())

# Sentiment Distribution
st.subheader("📌 Sentiment Distribution")
sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
st.plotly_chart(px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                       title='Sentiment Count', color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)

# Sentiment Over Time
st.subheader("📆 Sentiment Over Time")
time_series = filtered_df.groupby([filtered_df['Timestamp'].dt.date, 'Sentiment']).size().unstack().fillna(0)
st.line_chart(time_series)

# Sentiment by Location
if 'Location' in filtered_df.columns and filtered_df['Location'].notna().sum() > 0:
    st.subheader("🌍 Sentiment by Top Locations")
location_sentiment = filtered_df.groupby(['Location', 'Sentiment']).size().unstack().fillna(0)
# Ensure numeric values
location_sentiment = location_sentiment.apply(pd.to_numeric, errors='coerce').fillna(0)

# Get top locations by total tweets
location_sentiment['Total'] = location_sentiment.sum(axis=1)
top_locations = location_sentiment.sort_values(by='Total', ascending=False).drop(columns='Total').head(15)

# Reset index for long-form
top_locations_reset = top_locations.reset_index().melt(id_vars='Location', var_name='Sentiment', value_name='Count')

# Plot
fig = px.bar(top_locations_reset, x='Location', y='Count', color='Sentiment', barmode='stack',
             title="Sentiment in Top 15 Locations", color_discrete_sequence=px.colors.qualitative.Set2)

st.plotly_chart(fig, use_container_width=True)

# Export analyzed data
df.to_csv("CleanStream-Tweet.csv", index=False)
st.success("✅Analysis Complete Output Saved To:'CleanStream-Tweet.csv'.")
