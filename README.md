# Sh-25-DS-Project-RealTime-SentimentAnalysis-Of-Tweets.github.io
# 📊 Real-Time Sentiment Analysis of Tweets

This project is a complete **end-to-end data science solution** for analyzing real-time public sentiment on Twitter using NLP, translation, sentiment modeling, and interactive dashboards. It supports **multilingual tweets (Kannada, Hindi, English)** and enables live monitoring of sentiment trends.

## 📌 Project Objective

To build a system that:
- Collects real-time or offline tweets
- Cleans and preprocesses text
- Detects language and translates to English (if needed)
- Applies sentiment classification
- Visualizes sentiment by time, location, and language
- Allows real-time interaction through a web dashboard (Streamlit)

## 🌐 Technologies Used

- **Python
- **NLP**: TextBlob, langdetect
- **Data Viz**: Plotly, Matplotlib, Seaborn
- **Web App**: Streamlit
- **Tweet Collection**: `snscrape` (no Twitter API needed)
- **Power BI** (optional business dashboard)

## 🗃️ Dataset Details

- 📄 File: `RealTime-TweetDataset.csv`
- 📊 Tweets: 10,000+ synthetic & real tweets
- 🗣️ Languages: Kannada, Hindi, English
- 📍 Fields:
  - `Tweet_ID`, `Date_Time`, `User_Name`, `Tweet_Text`
  - `Language`, `Location`, `Sentiment`, `Topic`

You can also collect live tweets using the `tweet_scraper.py` script.

## 🚀 Features

- ✅ Multilingual tweet processing (supports regional Indian languages)
- ✅ Language detection using `langdetect`
- ✅ Automatic translation to English using `TextBlob`
- ✅ Sentiment classification (Positive / Negative / Neutral)
- ✅ Real-time and batch mode support
- ✅ Beautiful interactive dashboard using **Streamlit**
- ✅ Download filtered results as CSV
- ✅ Optional Power BI visualization

## 🧠 Folder Structure

📁 RealTime-Sentiment-Analysis/
│
├── RealTime-TweetDataset.csv        # Main dataset (or generated live)
├── Cleaned-RealTime-Tweet.csv       # Final processed output
├── RealTime-Tweet.py                # Main analysis logic
├── StreamTweet.py                   # Streamlit dashboard with live fetch
├── tweet_scraper.py                 # SNScrape-based live tweet fetcher
├── Streamlit App.gif                # Dashboard demo (optional)
├── *.png                            # Visuals used in report (daily trends, etc.)
├── TweetDashboard.pbix              # Optional Power BI version
└── README.md                        # This file
