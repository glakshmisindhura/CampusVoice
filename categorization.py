import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from nlp.preprocessing import clean_text


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "campusvoice_sample_dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "category_model.pkl"


def train_category_model():

    print("\nStarting Category Model Training...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Check required columns
    if not {"feedback", "category"}.issubset(df.columns):
        raise ValueError("Dataset must contain feedback and category columns.")

    # Remove missing values
    df = df.dropna(subset=["feedback", "category"])

    # Clean feedback
    df["cleaned_feedback"] = df["feedback"].apply(clean_text)

    # Remove empty feedback
    df = df[df["cleaned_feedback"] != ""]

    # Prepare training data
    X = df["cleaned_feedback"]

    y = df["category"].str.lower().str.strip()

    # Create machine learning pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(ngram_range=(1, 2))
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    # Train model using all labelled examples
    model.fit(X, y)

    # Create models folder
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nCategory Model Training Completed!")

    print("\nCategories learned:")

    for category in sorted(y.unique()):
        print("-", category)

    print("\nModel saved successfully!")

    print("Location:", MODEL_PATH)

    return model


def load_category_model():

    if not MODEL_PATH.exists():
        return train_category_model()

    return joblib.load(MODEL_PATH)


def predict_category(feedback):

    model = load_category_model()

    cleaned_feedback = clean_text(feedback)

    if not cleaned_feedback:
        return "other"

    prediction = model.predict([cleaned_feedback])

    return prediction[0]


if __name__ == "__main__":

    model = train_category_model()

    test_feedback = [
        "The hostel bathrooms are not clean",
        "The internet connection is very slow",
        "The library needs more books",
        "The canteen food is excellent",
        "The faculty explains topics clearly",
        "The college bus is always late"
    ]

    print("\nSample Category Predictions:")

    for feedback in test_feedback:

        result = predict_category(feedback)

        print("\nFeedback:", feedback)

        print("Predicted Category:", result) 