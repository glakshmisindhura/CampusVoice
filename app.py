from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import io
import os
from datetime import datetime

from nlp.preprocessing import clean_text
from nlp.sentiment import load_sentiment_model
from nlp.categorization import load_category_model
from nlp.issue_detection import detect_common_issues


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

DATASET_FOLDER = os.path.join(app.root_path, "dataset")
RECORDS_FILE = os.path.join(DATASET_FOLDER, "feedback_records.csv")

os.makedirs(DATASET_FOLDER, exist_ok=True)


# -----------------------------------
# NORMALIZE EXISTING FEEDBACK RECORDS
# -----------------------------------

def normalize_records(records):

    required_columns = [
        "feedback",
        "department",
        "cleaned_feedback",
        "sentiment",
        "category",
        "submitted_at",
        "timestamp"
    ]

    for column in required_columns:
        if column not in records.columns:
            records[column] = ""

    # Fix sentiment capitalization
    sentiment_mapping = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral"
    }

    records["sentiment"] = (
        records["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .apply(
            lambda value: sentiment_mapping.get(
                value.lower(), value
            )
        )
    )

    # Fix missing dates and support old records
    records["submitted_at"] = records["submitted_at"].fillna("")
    records["timestamp"] = records["timestamp"].fillna("")

    records["submitted_at"] = records.apply(
        lambda row: row["submitted_at"]
        if str(row["submitted_at"]).strip()
        else row["timestamp"],
        axis=1
    )

    records["timestamp"] = records["submitted_at"]

    records["department"] = (
        records["department"]
        .fillna("Not specified")
        .astype(str)
        .replace("", "Not specified")
    )

    return records[required_columns]


# -----------------------------------
# LOAD SAVED FEEDBACK
# -----------------------------------

def load_saved_feedback():

    if os.path.exists(RECORDS_FILE):

        try:
            records = pd.read_csv(
                RECORDS_FILE,
                encoding="utf-8-sig"
            )

            if records.empty:
                return normalize_records(records)

            return normalize_records(records)

        except pd.errors.EmptyDataError:
            pass

    return normalize_records(pd.DataFrame())


# -----------------------------------
# SAVE NEW FEEDBACK
# -----------------------------------

def save_feedback(new_records):

    existing_records = load_saved_feedback()

    new_records = normalize_records(new_records.copy())

    combined_records = pd.concat(
        [existing_records, new_records],
        ignore_index=True
    )

    combined_records = normalize_records(combined_records)

    combined_records.to_csv(
        RECORDS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# -----------------------------------
# NLP ANALYSIS
# -----------------------------------

def analyze_feedback(feedback_df):

    if "feedback" not in feedback_df.columns:
        raise ValueError(
            "CSV must contain a column named feedback."
        )

    feedback_df = feedback_df.dropna(
        subset=["feedback"]
    ).copy()

    feedback_df["feedback"] = (
        feedback_df["feedback"]
        .astype(str)
        .str.strip()
    )

    feedback_df = feedback_df[
        feedback_df["feedback"] != ""
    ].copy()

    if feedback_df.empty:
        raise ValueError("No valid feedback found.")

    if "department" not in feedback_df.columns:
        feedback_df["department"] = "Not specified"

    feedback_df["department"] = (
        feedback_df["department"]
        .fillna("Not specified")
        .astype(str)
        .str.strip()
        .replace("", "Not specified")
    )

    # Load NLP models
    sentiment_model = load_sentiment_model()
    category_model = load_category_model()

    # Clean feedback text
    feedback_df["cleaned_feedback"] = (
        feedback_df["feedback"].apply(clean_text)
    )

    # Predict sentiment
    feedback_df["sentiment"] = sentiment_model.predict(
        feedback_df["cleaned_feedback"]
    )

    # Predict category
    feedback_df["category"] = category_model.predict(
        feedback_df["cleaned_feedback"]
    )

    # Standardize sentiment labels
    sentiment_mapping = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral"
    }

    feedback_df["sentiment"] = (
        feedback_df["sentiment"]
        .astype(str)
        .str.strip()
        .apply(
            lambda value: sentiment_mapping.get(
                value.lower(), value
            )
        )
    )

    # Save date in both column names
    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    feedback_df["submitted_at"] = current_time
    feedback_df["timestamp"] = current_time

    return feedback_df


# -----------------------------------
# DASHBOARD DATA
# -----------------------------------

def get_dashboard_data():

    records = load_saved_feedback()

    # Always provide all three sentiment labels
    sentiment_counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    if records.empty:

        return {
            "results": [],
            "sentiment_counts": sentiment_counts,
            "category_counts": {},
            "common_issues": [],
            "total_feedback": 0
        }

    # Count sentiments
    actual_sentiment_counts = (
        records["sentiment"].value_counts().to_dict()
    )

    for sentiment in sentiment_counts:
        sentiment_counts[sentiment] = int(
            actual_sentiment_counts.get(sentiment, 0)
        )

    # Count categories
    category_counts = (
        records["category"]
        .replace("", "Uncategorized")
        .value_counts()
        .to_dict()
    )

    # Find negative feedback
    negative_feedback = records.loc[
        records["sentiment"] == "Negative",
        "feedback"
    ].tolist()

    # Detect common issues
    common_issues = detect_common_issues(
        negative_feedback
    ) if negative_feedback else []

    return {
        "results": records.to_dict(orient="records"),
        "sentiment_counts": sentiment_counts,
        "category_counts": category_counts,
        "common_issues": common_issues,
        "total_feedback": len(records)
    }


# -----------------------------------
# COMMON PAGE RENDER FUNCTION
# -----------------------------------

def show_page(page, error=None, success=None):

    dashboard = get_dashboard_data()

    return render_template(
        "index.html",
        page=page,
        **dashboard,
        analyzed=dashboard["total_feedback"] > 0,
        error=error,
        success=success
    )


# -----------------------------------
# 1. DASHBOARD
# -----------------------------------

@app.route("/")
def home():

    return show_page("dashboard")


# -----------------------------------
# 2. INDIVIDUAL FEEDBACK
# -----------------------------------

@app.route("/individual", methods=["GET", "POST"])
def individual_feedback():

    if request.method == "POST":

        try:
            feedback = request.form.get(
                "feedback", ""
            ).strip()

            department = request.form.get(
                "department", "Not specified"
            ).strip()

            if not feedback:
                raise ValueError("Please enter feedback.")

            feedback_df = pd.DataFrame([{
                "feedback": feedback,
                "department": department or "Not specified"
            }])

            analyzed_df = analyze_feedback(feedback_df)

            save_feedback(analyzed_df)

            return redirect(
                url_for("individual_feedback", saved="1")
            )

        except Exception as e:
            return show_page("individual", error=str(e))

    success = None

    if request.args.get("saved") == "1":
        success = "Feedback successfully analyzed and saved."

    return show_page("individual", success=success)


# -----------------------------------
# 3. BULK CSV UPLOAD
# -----------------------------------

@app.route("/bulk", methods=["GET", "POST"])
def bulk_upload():

    if request.method == "POST":

        try:
            uploaded_file = request.files.get("file")

            if not uploaded_file or not uploaded_file.filename:
                raise ValueError("Please select a CSV file.")

            if not uploaded_file.filename.lower().endswith(".csv"):
                raise ValueError("Please upload a CSV file.")

            feedback_df = pd.read_csv(uploaded_file)

            analyzed_df = analyze_feedback(feedback_df)

            count = len(analyzed_df)

            save_feedback(analyzed_df)

            return redirect(
                url_for("bulk_upload", saved=count)
            )

        except Exception as e:
            return show_page("bulk", error=str(e))

    success = None

    saved_count = request.args.get("saved")

    if saved_count:
        success = (
            f"{saved_count} feedback records successfully analyzed and saved."
        )

    return show_page("bulk", success=success)


# -----------------------------------
# 4. FEEDBACK RECORDS
# -----------------------------------

@app.route("/records")
def feedback_records():

    all_records = load_saved_feedback()

    department_filter = request.args.get("department", "")
    sentiment_filter = request.args.get("sentiment", "")
    category_filter = request.args.get("category", "")

    records = all_records.copy()

    if department_filter:
        records = records[
            records["department"] == department_filter
        ]

    if sentiment_filter:
        records = records[
            records["sentiment"] == sentiment_filter
        ]

    if category_filter:
        records = records[
            records["category"] == category_filter
        ]

    dashboard = get_dashboard_data()

    dashboard["results"] = records.to_dict(
        orient="records"
    )

    dashboard["departments"] = sorted(
        all_records["department"].dropna().unique().tolist()
    )

    dashboard["sentiments"] = sorted(
        all_records["sentiment"].dropna().unique().tolist()
    )

    dashboard["categories"] = sorted(
        all_records["category"].dropna().unique().tolist()
    )

    return render_template(
        "index.html",
        page="records",
        **dashboard,
        analyzed=dashboard["total_feedback"] > 0,
        error=None,
        success=None,
        selected_department=department_filter,
        selected_sentiment=sentiment_filter,
        selected_category=category_filter
    )


# -----------------------------------
# 5. CATEGORY ANALYSIS
# -----------------------------------

@app.route("/analysis")
def category_analysis():

    return show_page("analysis")


# -----------------------------------
# DOWNLOAD COMPLETE REPORT
# -----------------------------------

@app.route("/download")
def download_report():

    records = load_saved_feedback()

    if records.empty:
        return "No feedback records available to download.", 400

    output = io.BytesIO()

    records.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="campusvoice_complete_feedback_report.csv"
    )


# -----------------------------------
# FILE SIZE ERROR
# -----------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return "File too large. Maximum upload size is 5 MB.", 413


# -----------------------------------
# RUN APPLICATION
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)