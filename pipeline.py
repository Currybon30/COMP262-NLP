import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# --------------------------------------------------------------------------------
# 1. LOAD AND FLATTEN THE JSON DATA
# --------------------------------------------------------------------------------
def load_and_flatten(json_path: str, csv_output: str = None) -> pd.DataFrame:
    """
    Loads a JSON file (line-delimited) and flattens it into a Pandas DataFrame.

    Args:
        json_path (str): Path to the JSON file.
        csv_output (str, optional): If provided, saves the flattened CSV to this path.

    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    json_data = []
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data.append(json.loads(line))
    
    df = pd.json_normalize(json_data, sep='_')
    
    if csv_output is not None:
        df.to_csv(csv_output, index=False)
    
    return df


# --------------------------------------------------------------------------------
# 2. INITIAL EXPLORATION (OPTIONAL)
# --------------------------------------------------------------------------------
def initial_exploration(df: pd.DataFrame) -> None:
    """
    Performs basic data exploration: head, info, describe,
    and plots the number of reviews per product.

    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    print(df.head())
    print(df.info())
    print(df.describe())

    # Plot distribution of the number of reviews per product
    reviews_per_product = df['asin'].value_counts().reset_index()
    reviews_per_product.columns = ['product_id', 'num_reviews']
    reviews_per_product = reviews_per_product.sort_values(by='num_reviews', ascending=True)

    plt.figure(figsize=(6, 8))
    sns.barplot(x='num_reviews', y='product_id', data=reviews_per_product, palette="viridis")
    plt.title('Number of Reviews per Product')
    plt.xlabel('Number of reviews')
    plt.ylabel('Product ID')
    plt.yticks(fontsize=5)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------
# 3. BALANCING THE DATASET
# --------------------------------------------------------------------------------
def filter_and_downsample(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """
    Filters out products with fewer than `min_reviews` reviews,
    then down-samples each product to exactly `min_reviews` reviews.

    Args:
        df (pd.DataFrame): Input DataFrame with column "asin".
        min_reviews (int, optional): Minimum number of reviews per product.

    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    # Filter
    review_counts = df['asin'].value_counts()
    valid_products = review_counts[review_counts >= min_reviews].index
    df_filtered = df[df['asin'].isin(valid_products)]

    # Down-sample
    df_balanced = (
        df_filtered.groupby('asin', group_keys=False)
                   .apply(lambda x: x.sample(n=min_reviews, random_state=42))
    )
    return df_balanced


# --------------------------------------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------------------------------------
def remove_duplicates_and_empty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate reviews based on (reviewerID, asin, reviewText),
    then removes rows with NaN or empty reviewText.

    Args:
        df (pd.DataFrame): Input DataFrame with columns ["reviewerID", "asin", "reviewText"].

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates(subset=["reviewerID", "asin", "reviewText"], keep="first")

    # Remove NaN or empty reviewText
    df = df.dropna(subset=["reviewText"])
    df = df[df["reviewText"].str.strip() != ""]

    return df


def label_sentiment(rating: int) -> str:
    """
    Converts an integer rating to a sentiment label.

    Args:
        rating (int): The "overall" rating.

    Returns:
        str: "Positive", "Neutral", or "Negative".
    """
    if rating in [4, 5]: 
        return "Positive"
    elif rating == 3: 
        return "Neutral"
    elif rating in [1, 2]: 
        return "Negative"
    else: 
        return "Unknown"  # to handle unexpected values


def text_cleaning(df: pd.DataFrame, text_column: str = "reviewText") -> pd.DataFrame:
    """
    Lowercases text, removes punctuation, removes stopwords,
    tokenizes, and lemmatizes the text.

    Args:
        df (pd.DataFrame): Input DataFrame with a text column.
        text_column (str, optional): Name of the text column.

    Returns:
        pd.DataFrame: DataFrame with a cleaned text column.
    """
    # Ensure NLTK assets are downloaded
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    df[text_column] = df[text_column].str.lower()

    # Remove punctuation
    df[text_column] = df[text_column].str.replace(f"[{string.punctuation}]", "", regex=True)

    # Tokenize
    df[text_column] = df[text_column].apply(word_tokenize)

    # Remove stopwords & lemmatize
    df[text_column] = df[text_column].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    )

    return df


# --------------------------------------------------------------------------------
# 5. OUTLIER DETECTION & REMOVAL
# --------------------------------------------------------------------------------
def calculate_review_length(df: pd.DataFrame, text_column: str = "reviewText") -> pd.DataFrame:
    """
    Adds a 'review_length' column with the character length of the text.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.

    Returns:
        pd.DataFrame: DataFrame with new 'review_length' column.
    """
    df["review_length"] = df[text_column].apply(lambda x: len(str(x)))
    return df


def detect_outliers_iqr(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detects outliers based on the IQR rule (1.5 * IQR above the 75th percentile
    or below the 25th percentile).

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): The column to check for outliers.

    Returns:
        pd.DataFrame: Subset of rows that are outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


# --------------------------------------------------------------------------------
# 6. SENTIMENT ANALYSIS (VADER & TEXTBLOB)
# --------------------------------------------------------------------------------
def apply_vader(df: pd.DataFrame, text_column: str = "reviewText") -> pd.DataFrame:
    """
    Applies VADER sentiment analysis to a specified text column in a DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the text data.
        text_column (str, optional): The column name containing the text to analyze.

    Returns:
        pd.DataFrame: The DataFrame with new columns for VADER sentiment scores.
    """
    analyzer = SentimentIntensityAnalyzer()

    # Convert token lists back to strings for VADER or it will interpret them incorrectly
    # If you want to keep tokens for other tasks, store them separately or convert on-the-fly.
    df_temp = df.copy()
    df_temp['reviewText_str'] = df_temp[text_column].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else tokens)

    df_temp['vader_neg'] = df_temp['reviewText_str'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df_temp['vader_neu'] = df_temp['reviewText_str'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    df_temp['vader_pos'] = df_temp['reviewText_str'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df_temp['vader_compound'] = df_temp['reviewText_str'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Drop the temporary string column if you want
    df_temp.drop(columns=["reviewText_str"], inplace=True)
    return df_temp


def apply_textblob(df: pd.DataFrame, text_column: str = "reviewText") -> pd.DataFrame:
    """
    Applies TextBlob sentiment analysis to a specified text column in a DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the text data.
        text_column (str, optional): The column name containing the text to analyze.

    Returns:
        pd.DataFrame: The DataFrame with new columns for TextBlob sentiment scores.
    """
    # Similar handling for token vs. string
    df_temp = df.copy()
    df_temp['reviewText_str'] = df_temp[text_column].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else tokens)

    df_temp['textblob_polarity'] = df_temp['reviewText_str'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_temp['textblob_subjectivity'] = df_temp['reviewText_str'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    df_temp.drop(columns=["reviewText_str"], inplace=True)
    return df_temp


# --------------------------------------------------------------------------------
# 7. PUTTING IT ALL TOGETHER
# --------------------------------------------------------------------------------
def main_pipeline(json_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    Orchestrates the entire NLP pipeline, from loading data to sentiment analysis.

    Args:
        json_path (str): Path to the raw JSON file.
        output_csv (str, optional): If provided, saves the final DataFrame to CSV.

    Returns:
        pd.DataFrame: The final cleaned and enriched DataFrame.
    """
    # Step 1: Load & Flatten
    df = load_and_flatten(json_path)

    # Step 2: (Optional) Basic Exploration
    # initial_exploration(df)  # Uncomment to see EDA details

    # Step 3: Balancing (Filter and Downsample)
    df = filter_and_downsample(df, min_reviews=5)

    # Step 4: Cleaning
    df = remove_duplicates_and_empty(df)

    # Add a "sentiment" column based on star rating
    df["sentiment"] = df["overall"].apply(label_sentiment)

    # Clean Text (lowercase, punctuation removal, stopwords, lemmatize)
    df = text_cleaning(df, text_column="reviewText")

    # Step 5: Outlier Detection and Removal
    df = calculate_review_length(df, text_column="reviewText")
    outliers = detect_outliers_iqr(df, "review_length")
    df = df[~df.index.isin(outliers.index)]

    # Step 6: Sentiment Analysis
    df = apply_vader(df, text_column="reviewText")
    df = apply_textblob(df, text_column="reviewText")

    # Step 7: Save final data (optional)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Final DataFrame saved to {output_csv}")

    return df


# --------------------------------------------------------------------------------
# USAGE EXAMPLE (if this is a standalone script, wrap in `if __name__ == "__main__":`)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # You can point this to your Amazon Appliance_5 JSON file, for example:
    JSON_INPUT_FILE = "Appliances_5.json"
    OUTPUT_CSV_FILE = "final_reviews_with_sentiment.csv"

    final_df = main_pipeline(JSON_INPUT_FILE, output_csv=OUTPUT_CSV_FILE)

    print("Pipeline completed. Here are the first rows of the final DataFrame:")
    print(final_df.head(10))
