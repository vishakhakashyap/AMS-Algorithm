import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter

# Load the dataset (replace 'twitter_data.csv' with your dataset file)
data = pd.read_csv('twitter_data.csv')

# Sentiment analysis using TextBlob
data['Sentiment'] = data['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Streamlit UI
st.title('Twitter Sentiment Analysis')
st.write("## Sentiment Analysis")
st.write("This app performs sentiment analysis on a Twitter dataset and visualizes the results.")

# Display the sentiment distribution using Matplotlib
st.write("### Sentiment Distribution")
st.write("The following plot shows the distribution of sentiment polarity in the dataset.")
fig_sentiment = plt.figure(figsize=(8, 6))
plt.hist(data['Sentiment'], bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Density')
st.pyplot(fig_sentiment)

# Analyze word frequency and create a word cloud
all_tweets = " ".join(data['tweet'])

# Count word frequencies
word_freq = Counter(all_tweets.split())
top_words = word_freq.most_common(10)

# Display the top words in a bar chart using Matplotlib
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
st.write("### Top 10 Most Frequent Words")
st.write("The following bar chart shows the top 10 most frequent words in the dataset.")
fig_wordcloud = plt.figure(figsize=(10, 6))
plt.barh(top_words_df['Word'], top_words_df['Frequency'], color='b')
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
st.pyplot(fig_wordcloud)
