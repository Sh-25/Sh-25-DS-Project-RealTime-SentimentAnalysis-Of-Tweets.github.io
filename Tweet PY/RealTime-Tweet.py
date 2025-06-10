import pandas as pd
import re
from langdetect import detect
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
from transformers import pipeline

# Step 1: Read the CSV file
df = pd.read_csv("DSProject-TWEET/RealTime-TweetDataset.csv") 
print(df)
print(df.columns)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Preprocessing (simple version)
#check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

#check for duplicate values
print("Duplicate values before cleaning:")
print(df.duplicated().sum())

#describe the dataset
print("Dataset description:")
print(df.describe())

#info about the dataset
print("Dataset info:")
print(df.info())

# Step 2: Preprocess Text and Detect Language
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Dictionary mapping ISO codes to full names
lang_map = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'af': 'unknown',
    'unknown': 'Unknown'
}

df['cleaned_text'] = df['Tweet_text'].astype(str).apply(clean_text)
df['Language'] = df['cleaned_text'].apply(lambda x: detect(x) if x.strip() else 'unknown')
# Apply mapping
df['Language'] = df['Language'].map(lang_map).fillna(df['Language'])

## Save the output to a CSV file
df.to_csv("DSProject-TWEET/Cleaned-RealTime-Tweet.csv", index=False)
print("Cleaned  Complete. Output saved to ",'Cleaned-RealTime-Tweet')

#Step 3: Sentiment Classification Using Pre-trained Transformer
sentiment_analyzer = pipeline("sentiment-analysis")

# List of example texts
texts = [
    "I love this product! It's amazing.",
    "This is terrible. I hate it.",
    "Not bad, could be better."
]

# Analyze each text
for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.2f}\n")

#Step 4: Analyze and Visualize Trends Over Time
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['date'] = df['Timestamp'].dt.date

daily_sentiment = df.groupby(['date', 'Sentiment']).size().unstack().fillna(0)

daily_sentiment.plot(figsize=(12, 6), marker='o')
plt.title("Daily Sentiment Trend")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Visualize Language Distribution
language_counts = df['Language'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values, palette='Pastel1', edgecolor='black')
plt.title("Language Distribution of Tweets")
plt.xlabel("Language")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Visualize Sentiment Distribution
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set1', edgecolor='black')
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
