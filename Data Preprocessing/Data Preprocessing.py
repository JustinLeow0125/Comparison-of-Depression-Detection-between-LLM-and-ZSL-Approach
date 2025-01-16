# ==================== Import Necessary Libraries ====================

import pandas as pd  # For handling tabular data (DataFrame operations)
import numpy as np  # For numerical operations and statistics
import re  # For regular expressions to clean text
import os  # For operating system-dependent functionality (e.g., checking file existence)
from sklearn.preprocessing import StandardScaler  # For feature scaling (if needed later)
import nltk  # Natural Language Toolkit, for text processing tasks
from nltk.corpus import stopwords  # For stopword removal
from nltk.stem import WordNetLemmatizer  # For lemmatizing words to their base form
# from imblearn.over_sampling import SMOTE  # Uncomment if you plan to use SMOTE later for handling class imbalance
from sklearn.model_selection import KFold, train_test_split  # For creating folds and splitting the dataset
from wordcloud import WordCloud  # For visualizing frequent words in a dataset
import matplotlib.pyplot as plt  # For plotting visualizations
import emoji  # For handling and removing emojis
from nltk.tokenize import word_tokenize  # For tokenizing text into words
from bs4 import BeautifulSoup  # Optional: For parsing HTML/XML (advanced cleaning if needed)

# ------------------------------
# Step 0: Initial Setup
# ------------------------------

# Ensure reproducibility by setting seeds across libraries
import random
import torch

def set_seed(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)  # Python's built-in random seed
    np.random.seed(seed)  # NumPy's random seed
    torch.manual_seed(seed)  # PyTorch CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed if CUDA is available

# Set the seed for reproducibility
set_seed(42)

# ------------------------------
# Step 1: Download Necessary NLTK Data Files
# ------------------------------

# Download stopwords list, lemmatizer data, and tokenizers needed for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# ------------------------------
# Step 2: Load Your Dataset
# ------------------------------

# Path to the dataset file
file_path = r'C:\Dataset\combined_depression_dataset.csv'

try:
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} records.")
except FileNotFoundError:
    # If the file does not exist, print an error and exit
    print(f"Error: The file {file_path} does not exist.")
    exit()
except pd.errors.EmptyDataError:
    # If the file is empty, print an error and exit
    print(f"Error: The file {file_path} is empty.")
    exit()
except Exception as e:
    # Catch-all for unexpected errors
    print(f"An unexpected error occurred while loading the dataset: {e}")
    exit()

# ------------------------------
# Step 3: Focus on the Relevant Column
# ------------------------------

column_name = "message to examine"

# Check if the required column is present in the DataFrame
if column_name not in df.columns:
    print(f"Error: The column '{column_name}' does not exist in the dataset.")
    exit()

# ------------------------------
# Step 4: Data Cleaning and Preprocessing
# ------------------------------

# Step 4.1: Handling Missing Values
# Replace any missing values in the target column with an empty string
df[column_name] = df[column_name].fillna('')

# Step 4.2: Removing Duplicate Records
# Drop duplicate rows based on the target column
df.drop_duplicates(subset=[column_name], inplace=True)
print(f"After removing duplicates: {df.shape[0]} records.")

# Step 4.3: Data Consistency
# Convert text to lowercase and strip leading/trailing whitespaces
df[column_name] = df[column_name].str.lower().str.strip()

# Step 4.4: Comprehensive PII Handling - Remove/Anonymize PII
def remove_pii(text):
    """
    Removes or replaces Personally Identifiable Information (PII) such as UIDs, emails, phone numbers, and URLs.
    """
    if not isinstance(text, str):
        return text

    # Remove UIDs (e.g., uid1234)
    text = re.sub(r'\buid\d+\b', '', text)

    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove phone numbers (assuming 10-15 digit phone numbers)
    text = re.sub(r'\b\d{10,15}\b', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove full names (Very simplistic pattern; can be improved with NER)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '', text)

    return text

df[column_name] = df[column_name].apply(remove_pii)

# Step 4.5: Removal of XML and Metadata Artifacts
def remove_unwanted_artifacts(text):
    """
    Removes XML-like tags, MS Word artifacts, and other unwanted technical terms from the text.
    """
    if not isinstance(text, str):
        return text
    
    # Remove XML-like tags
    text = re.sub(r'<[^>]+>', '', text)

    # List of unwanted patterns to remove
    unwanted_patterns = [
        "wlsdexception", "lockedfalse", "priority", "semihidden", "name", "grid", "accent", 
        "list", "tablestyle", "latents", "font", "shading", "paragraph", "mso", "style", "normal",
        "xml", "w:", "m:", "o:", "p:", "r:", "s:", "t:", "xmlns", "docprops", "doctype", "html",
        "endif", "if", "gte", "lang", "align", "center", "class", "table", "td", "tr", "div", 
        "span", "p", "charset", "utf", "false", "true", "deflockedstate", "defunhidewhenused",
        "defsemihidden", "defqformat", "defpriority", "latentstylecount"
    ]

    # Remove each unwanted pattern
    for pattern in unwanted_patterns:
        text = re.sub(r'\b' + re.escape(pattern) + r'\b', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df[column_name] = df[column_name].apply(remove_unwanted_artifacts)

# Step 4.6: Removal of Punctuation and Special Characters
# Keep only alphabets and whitespace
df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)

# Step 4.7: Removing Repeated Characters
# Limit sequences of the same character to two occurrences (e.g., "heyyy" -> "heyy")
df[column_name] = df[column_name].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1\1', x) if isinstance(x, str) else x)

# Step 4.8: Removal of Emojis and Emoticons
def remove_emojis(text):
    """
    Removes all emojis from the text.
    """
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) else text

df[column_name] = df[column_name].apply(remove_emojis)

# Step 4.9: Tokenization
# Split the cleaned text into words (tokens)
df['tokens'] = df[column_name].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])

# Step 4.10: Handling Negations
def handle_negations(tokens):
    """
    Handles negations by appending a 'NOT_' prefix to the word that follows a negation word.
    This helps capture negated meaning in sentiment analysis.
    """
    negations = {"not", "no", "never", "n't", "cannot", "can't", "won't", "don't"}
    new_tokens = []
    negate = False
    for word in tokens:
        if word in negations:
            # If we hit a negation word, set negate to True
            negate = True
            new_tokens.append(word)
        elif negate:
            # Prefix the next word with NOT_
            new_tokens.append(f"NOT_{word}")
            negate = False
        else:
            new_tokens.append(word)
    return new_tokens

df['tokens'] = df['tokens'].apply(handle_negations)

# Step 4.11: Removal of Stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Step 4.12: Lemmatization
# Convert words to their base form (e.g., "running" -> "run")
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Step 4.13: Join Tokens Back to Text
# Reconstruct the processed text from tokens
df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

# Step 4.14: Replace Original Column and Drop Intermediate Columns
df[column_name] = df['processed_text']
df.drop(columns=['tokens', 'processed_text'], inplace=True)

# Step 4.15: Feature Engineering
# Create additional features from the cleaned text

# 1. Text Length (Number of characters)
df['text_length'] = df[column_name].apply(len)

# 2. Word Count
df['word_count'] = df[column_name].apply(lambda x: len(x.split()))

# 3. Average Word Length
df['avg_word_length'] = df[column_name].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)

# 4. Count of Stopwords (in the final processed text)
def count_stopwords(text):
    if isinstance(text, str):
        words = word_tokenize(text)
        return len([word for word in words if word in stop_words])
    return 0

df['stopword_count'] = df[column_name].apply(count_stopwords)

def generate_wordcloud(text, title):
    """
    Generates and displays a word cloud from the given text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

# If a 'label' column exists, we can generate word clouds for each class
if 'label' in df.columns:
    depressed_text = ' '.join(df[df['label'] == 1][column_name])
    non_depressed_text = ' '.join(df[df['label'] == 0][column_name])

    generate_wordcloud(depressed_text, "Word Cloud for Depressed Messages")
    generate_wordcloud(non_depressed_text, "Word Cloud for Non-Depressed Messages")
else:
    print("Warning: 'label' column not found. Skipping word cloud generation.")

output_file_path = r'C:\Dataset\Depression Analysis Dataset (DAD).csv'

try:
    df.to_csv(output_file_path, index=False)
    print(f"Final processed file successfully saved to: {output_file_path}")
except PermissionError as e:
    print(f"Permission Error: {e}. Please ensure the directory exists and you have write permissions.")
except Exception as e:
    print(f"An unexpected error occurred while saving the final file: {e}")

# Show the first few rows of the final processed DataFrame
print("\nProcessed DataFrame Preview:")
print(df.head())

# Check if no placeholders for PII remain
pii_placeholders = ['<UID>', '<EMAIL>', '<PHONE>', '<NAME>']
for placeholder in pii_placeholders:
    if df[column_name].str.contains(placeholder).any():
        print(f"Warning: Placeholder {placeholder} still present in the data.")
    else:
        print(f"Placeholder {placeholder} successfully removed.")

# Display statistical summary of engineered features
print("\nStatistical Summary of Engineered Features:")
print(df[['text_length', 'word_count', 'avg_word_length', 'stopword_count']].describe())

# Plot histograms of the engineered features
df[['text_length', 'word_count', 'avg_word_length', 'stopword_count']].hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.show()
