from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from nlp.preprocessing import clean_text


def detect_common_issues(feedback_list, top_n=10):
    """
    Extract frequently occurring words and phrases
    from negative student feedback.
    """

    # Clean all feedback
    cleaned_feedback = [
        clean_text(feedback)
        for feedback in feedback_list
        if isinstance(feedback, str) and clean_text(feedback)
    ]

    if not cleaned_feedback:
        return []

    # Convert text into word and phrase frequencies
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100
    )

    try:
        matrix = vectorizer.fit_transform(cleaned_feedback)
    except ValueError:
        return []

    # Calculate keyword frequencies
    keywords = vectorizer.get_feature_names_out()

    frequencies = matrix.sum(axis=0).A1

    # Store keyword counts
    issue_counts = defaultdict(int)

    for keyword, count in zip(keywords, frequencies):
        issue_counts[keyword] = int(count)

    # Sort by frequency
    sorted_issues = sorted(
        issue_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Return top recurring issues
    results = []

    for issue, count in sorted_issues[:top_n]:

        results.append({
            "issue": issue,
            "count": count
        })

    return results


if __name__ == "__main__":

    sample_feedback = [
        "The hostel Wi-Fi is very slow.",
        "The hostel Wi-Fi disconnects frequently.",
        "The internet connection is very slow.",
        "The hostel bathrooms are not clean.",
        "The canteen food is very poor.",
        "The hostel Wi-Fi is not working."
    ]

    issues = detect_common_issues(sample_feedback)

    print("\nCampusVoice Common Issue Detection")

    print("----------------------------------")

    for item in issues:

        print(
            "Issue:",
            item["issue"],
            "| Frequency:",
            item["count"]
        )