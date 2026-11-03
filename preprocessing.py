import re


def clean_text(text):
    """
    Clean student feedback before NLP analysis.
    """

    if not isinstance(text, str):
        return ""

    # Convert text into lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters but preserve letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    Split cleaned feedback into individual words.
    """

    cleaned_text = clean_text(text)

    words = cleaned_text.split()

    return words


if __name__ == "__main__":

    sample = "The HOSTEL Wi-Fi is VERY slow!!!"

    print("Original Feedback:", sample)

    print("Cleaned Feedback:", clean_text(sample))

    print("Tokens:", tokenize_text(sample)) 