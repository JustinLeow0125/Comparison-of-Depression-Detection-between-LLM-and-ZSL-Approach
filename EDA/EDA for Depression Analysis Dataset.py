import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
file_path = r"C:\Dataset\Depression Analysis Dataset (DAD).csv"
df = pd.read_csv(file_path)

# Verify dataset columns
print(df.head())

# Ensure the column 'message to examine' is properly named
df.rename(columns={'message to examine': 'post'}, inplace=True)

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Apply cleaning function to 'post' column
df['clean_post'] = df['post'].dropna().apply(clean_text)

# Separate depressed and non-depressed messages
depressed_posts = df[df['label'] == 1]['clean_post'].dropna()
non_depressed_posts = df[df['label'] == 0]['clean_post'].dropna()

# Word Cloud for Depressed Messages
depressed_text = " ".join(depressed_posts)
wordcloud_depressed = WordCloud(width=800, height=400, background_color='white').generate(depressed_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_depressed, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Depressed Messages')
plt.show()

# Word Cloud for Non-Depressed Messages
non_depressed_text = " ".join(non_depressed_posts)
wordcloud_non_depressed = WordCloud(width=800, height=400, background_color='white').generate(non_depressed_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_non_depressed, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Non-Depressed Messages')
plt.show()

# Plot Distributions for text_length, word_count, avg_word_length, and stopword_count
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Text Metrics Distributions')

# Text Length Distribution
sns.histplot(df['text_length'], bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Text Length')
axes[0, 0].grid(True)

# Word Count Distribution
sns.histplot(df['word_count'], bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Word Count')
axes[0, 1].grid(True)

# Average Word Length Distribution
sns.histplot(df['avg_word_length'], bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Average Word Length')
axes[1, 0].grid(True)

# Stopword Count Distribution
sns.histplot(df['stopword_count'], bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Stopword Count')
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
