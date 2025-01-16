import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')  # For lemmatization (optional)
nltk.download('omw-1.4')  # For lemmatization (optional)

from nltk.stem import WordNetLemmatizer

# 1. Load the Dataset
file_path = r"C:\Dataset\sentiment_tweets3.csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8')  # Adjust encoding if necessary
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found at the path: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Display column names and first few rows to verify
print("Column Names:", df.columns.tolist())
print(df.head())

# Assign column names to variables for easier reference
index_col = 'Index'  # Assuming 'Index' is the name
message_col = 'message to examine'  # Updated to match your data
label_col = 'label (depression result)'  # Updated to match your data

# Convert label column to string to ensure consistency
df[label_col] = df[label_col].astype(str)

# Verify unique labels and their types
unique_labels = df[label_col].unique()
print("Unique Labels:", unique_labels)
print("Label Column Type:", df[label_col].dtype)

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Optionally, drop rows with missing values in key columns
df.dropna(subset=[message_col, label_col], inplace=True)

# 2. Distribution of Depression Labels (Bar Chart)
plt.figure(figsize=(8,6))

# Create a palette using 'viridis' with as many colors as unique labels
palette = sns.color_palette("viridis", n_colors=len(unique_labels))

# Create a mapping from label to color
label_colors = dict(zip(unique_labels, palette))

# Print label_colors to verify
print("Label Colors Mapping:", label_colors)

# Plot the countplot with the custom palette
sns.countplot(data=df, x=label_col, palette=label_colors)

plt.title('Distribution of Depression Labels')
plt.xlabel('Depression Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribution of Message Lengths
# Calculate message lengths
df['message_length'] = df[message_col].astype(str).apply(len)

plt.figure(figsize=(10,6))
sns.histplot(df['message_length'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Message Lengths')
plt.xlabel('Message Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Word Cloud of All Messages
# Combine all messages into one text
all_messages = ' '.join(df[message_col].astype(str).tolist())

# Define stopwords
stop_words = set(stopwords.words('english'))
additional_stopwords = set(['https', 'http', 'co', 'RT', 'add', 'me', 'on', 'myspace'])  # Added more specific stopwords
stop_words = stop_words.union(additional_stopwords)

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
all_messages_clean = all_messages.translate(translator)

# Optional Preprocessing Steps:
# Lowercase
all_messages_clean = all_messages_clean.lower()

# Remove numbers
all_messages_clean = ''.join([char for char in all_messages_clean if not char.isdigit()])

# Lemmatization (Optional)
lemmatizer = WordNetLemmatizer()
all_messages_tokens = all_messages_clean.split()
all_messages_lemmatized = ' '.join([lemmatizer.lemmatize(word) for word in all_messages_tokens])

# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stop_words,
    max_words=200,
    max_font_size=100,
    random_state=42
).generate(all_messages_lemmatized)

# Plot the Word Cloud
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of All Messages', fontsize=20)
plt.tight_layout(pad=0)
plt.show()
