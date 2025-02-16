import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'amazon.csv'  # Define the file path
df = pd.read_csv(file_path)  # Load the CSV into a DataFrame
print(df.shape)

# Preprocess the review text (basic cleaning)
def clean_text(text):
    return re.sub(r"[^a-zA-Z]", ' ', str(text)).lower()

df['reviewText'] = df['reviewText'].apply(clean_text)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to determine sentiment category
def sentiment_category(text):
    if pd.isnull(text):  # Handle NaN values
        return "Neutral"
    scores = analyzer.polarity_scores(text)
    neg, pos = scores['neg'], scores['pos']
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

# Apply the sentiment analysis function
df['sentiment'] = df['reviewText'].apply(sentiment_category)

# See the analysis
df[['reviewText', 'sentiment']].to_csv('review_sentiments.csv', index=False)

print("Sentiments saved to 'review_sentiments.csv'")

# Calculate sentiment percentages
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100

positive_percentage = sentiment_counts.get("Positive", 0)
negative_percentage = sentiment_counts.get("Negative", 0)
neutral_percentage = sentiment_counts.get("Neutral", 0)

# Add VADER compound score for each review
def calculate_compound(text):
    if pd.isnull(text):  # Handle NaN values
        return 0
    return analyzer.polarity_scores(text)['compound']

df['compound'] = df['reviewText'].apply(calculate_compound)

# Calculate the confidence score
confidence_score = (df['compound'].mean() + 1) / 2 * 10  # Scale compound score to 0-10

# Display results
print("SENTIMENT DISTRIBUTION".center(50, '-'))
print(f"Positive: {positive_percentage:.2f}%")
print(f"Negative: {negative_percentage:.2f}%")
print(f"Neutral: {neutral_percentage:.2f}%")

print("CONFIDENCE SCORE".center(50, '-'))
print(f"The overall confidence score for the product is: {confidence_score:.2f} out of 10")


# Visualization: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')  # Save the plot as an image
plt.show()

# Visualization: Compound Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['compound'], bins=20, kde=True, color='blue')
plt.title('Compound Score Distribution', fontsize=16)
plt.xlabel('Compound Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('compound_score_distribution.png')  # Save the plot as an image
plt.show()