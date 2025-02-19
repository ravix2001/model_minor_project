import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained RandomForest model
with open('model_RandomForestClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load new dataset
df = pd.read_csv('reviews.csv')

# Handle missing values in the reviewText column
df['reviewText'] = df['reviewText'].fillna('')  # Replace NaN with an empty string

# Transform reviews using the saved vectorizer
X_test = vectorizer.transform(df['reviewText'])

# Make predictions
predictions = model.predict(X_test)

# **Manually map numeric predictions to sentiment labels**
sentiment_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Apply mapping to predictions
df['sentiment'] = [sentiment_mapping[pred] for pred in predictions]

# Calculate the percentage of each sentiment label
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100

positive_percentage = sentiment_counts.get("Positive", 0)
negative_percentage = sentiment_counts.get("Negative", 0)
neutral_percentage = sentiment_counts.get("Neutral", 0)

# Mapping sentiment labels to numerical scores
sentiment_scores = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1
}

# Function to calculate compound score based on sentiment labels
def get_compound_score(sentiment):
    return sentiment_scores.get(sentiment, 0)  # Default to 0 if sentiment is missing

# Apply the function to calculate compound scores for each review
df['compound'] = df['sentiment'].apply(get_compound_score)

# Save the results to a new CSV file
df[['reviewText', 'sentiment', 'compound']].to_csv('review_with_compound_scores.csv', index=False)

print("Compound scores calculated and saved!")

# Calculate the overall compound score
compound_score = (df['compound'].mean()) * 10  # Scale compound score to 0-10

# Display results
print("SENTIMENT DISTRIBUTION".center(50, '-'))
print(f"Positive: {positive_percentage:.2f}%")
print(f"Negative: {negative_percentage:.2f}%")
print(f"Neutral: {neutral_percentage:.2f}%")

print("CONFIDENCE SCORE".center(50, '-'))
# Print the overall compound score
print(f"Overall Compound Score for the product is: {compound_score:.2f} out of 10")

# Visualization: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, palette='viridis', legend=False)
plt.title('Sentiment Distribution', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')  # Save the plot as an image
plt.show()