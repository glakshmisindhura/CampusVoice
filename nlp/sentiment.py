import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from nlp.preprocessing import clean_text


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "campusvoice_sample_dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"


def train_sentiment_model():

    print("\nStarting Sentiment Model Training...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Check required columns
    if not {"feedback", "sentiment"}.issubset(df.columns):
        raise ValueError("Dataset must contain feedback and sentiment columns.")

    # Remove missing values
    df = df.dropna(subset=["feedback", "sentiment"])

    # Clean feedback
    df["cleaned_feedback"] = df["feedback"].apply(clean_text)

    # Remove empty feedback
    df = df[df["cleaned_feedback"] != ""]

    # Prepare input and output
    X = df["cleaned_feedback"]
    y = df["sentiment"].str.lower().str.strip()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create NLP machine learning pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Training Completed!")

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Create models folder if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)

    return model


def load_sentiment_model():

    if not MODEL_PATH.exists():
        return train_sentiment_model()

    return joblib.load(MODEL_PATH)


def predict_sentiment(feedback):

    model = load_sentiment_model()

    cleaned_feedback = clean_text(feedback)

    if not cleaned_feedback:
        return "neutral"

    prediction = model.predict([cleaned_feedback])

    return prediction[0]


if __name__ == "__main__":

    model = train_sentiment_model()

    test_feedback = [
        "The faculty teaching is excellent",
        "The hostel maintenance is very poor",
        "The library opens at 9 AM"
    ]

    print("\nSample Sentiment Predictions:")

    for feedback in test_feedback:

        result = predict_sentiment(feedback)

        print("\nFeedback:", feedback)
        print("Sentiment:", result) 