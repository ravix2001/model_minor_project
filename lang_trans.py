import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect

# Load the dataset
df = pd.read_csv("reviews.csv")

# Function to translate text to English
def translate_to_english(text):
    try:
        lang = detect(text)  # Detect language
        if lang != "en":  # If not English, translate
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text  # If already English, return as is
    except:
        return text  # If detection fails, return original

# Apply translation to reviewText column
df["translated_review"] = df["reviewText"].astype(str).apply(translate_to_english)

# Save the translated dataset
df.to_csv("translated_reviews.csv", index=False)

print("Translation completed and saved!")
