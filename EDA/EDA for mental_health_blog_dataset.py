import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
file_path = r'C:\Users\notth\OneDrive - Sunway Education Group\Documents\Capstone 2\Dataset\mental_health_blog_dataset.csv'
df = pd.read_csv(file_path)

# Clean the column names (remove leading/trailing spaces and convert to snake_case)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Print column names to check for issues
print("Column Names:", df.columns)

# Basic Information about the Dataset
print("\nDataset Info:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Analyze Categories
category_counts = df['category'].value_counts()
category_counts.plot(kind='bar', title='Post Counts by Category')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

# Posts Over Time
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, handling errors
posts_by_date = df.groupby(df['date'].dt.to_period('M')).size()
posts_by_date.plot(kind='line', title='Number of Posts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.show()

# Generate Word Cloud for Posts
text = ' '.join(df['post'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Posts')
plt.show()
