import re


def preprocess_text(text):
    # Remove links/urls
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    # Expand contracted words
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "don't": "do not",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "they're": "they are",
        "we're": "we are",
        "I've": "I have",
        "you've": "you have",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "wasn't": "was not",
        "hadn't": "had not",
        "mightn't": "might not",
        "isn't": "is not",
        "aren't": "are not",
        "let's": "let us",
        "there's": "there is",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)  # check
    # Remove jargon words like @quot and #SXSW
    text = re.sub(r"&quot;|#SXSW", "", text, flags=re.IGNORECASE)
    # Remove hyphen between words
    text = text.replace("-", " ")
    # Lowercase
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    return text


def sanitize_output(text):
    category = text.split("\n")[0]
    # category = re.search(r"`(.*?)`", text).group(1)
    return category


def process_labels(df, keys, mappings):
    df = df.copy()
    if not isinstance(keys, list):
        keys = [keys]
    if not isinstance(mappings, list):
        mappings = [mappings]

    for key, mapping in zip(keys, mappings):
        df.loc[:, f"{key}_label"] = df[key].map(mapping)
    return df
