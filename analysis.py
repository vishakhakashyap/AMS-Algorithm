import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter

# Load the dataset (replace 'twitter_data.csv' with your dataset file)
data = pd.read_csv('twitter_data.csv')

# Sentiment analysis using TextBlob
data['Sentiment'] = data['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Create a bar chart to visualize the sentiment distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Sentiment'], bins=30, kde=True)
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.show()

# Analyze word frequency and create a word cloud
all_tweets = " ".join(data['tweet'])
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(all_tweets)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()

# Count word frequencies
word_freq = Counter(" ".join(data['tweet']).split())
top_words = word_freq.most_common(10)

# Create a bar chart for the top words
words, freq = zip(*top_words)
plt.figure(figsize=(10, 6))
sns.barplot(x=freq, y=words, palette='viridis')
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()