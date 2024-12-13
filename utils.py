import re
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict

stop_words = set(stopwords.words("english"))

# Define regex patterns
title_modelword_pattern = re.compile(
    r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
)
value_modelword_pattern = re.compile(r"(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)")

tv_brands = [
    "Samsung",
    "LG",
    "Sony",
    "TCL",
    "Hisense",
    "Vizio",
    "Panasonic",
    "Philips",
    "Sharp",
    "Toshiba",
    "Insignia",
    "Roku",
    "JVC",
    "Westinghouse",
    "Hitachi",
    "Sanyo",
    "Element",
    "Magnavox",
    "Proscan",
    "Skyworth",
    "Funai",
    "Aiwa",
    "Pioneer",
    "Konka",
    "Haier",
    "BenQ",
    "Mitsubishi",
    "RCA",
    "Xiaomi",
    "Changhong",
    "Loewe",
    "Grundig",
    "Seiki",
    "Blaupunkt",
    "Polaroid",
    "Metz",
    "Bush",
    "Onida",
    "Sansui",
    "Akai",
    "Thomson",
    "BPL",
    "Tatung",
    "Eizo",
    "ViewSonic",
    "Croma",
    "Hannspree",
    "Vestel",
    "Nordmende",
    "Hitense",
    "Videocon",
    "NEC",
    "Sharp",
]

# Create a lowercase version of the brands to make case-insensitive matches easier
tv_brands_lower = [brand.lower() for brand in tv_brands]


def get_brand(text):
    brand = []
    text = text.lower()
    words = re.split(r"\s|-", text)
    for tv_brand in tv_brands_lower:
        if tv_brand in words:
            brand.append(tv_brand)
    return brand


def extract_model_words_from_title(text):
    if not isinstance(text, str):
        return []
    # Find all matches for title model words
    candidates = title_modelword_pattern.findall(text)
    # The regex with findall returns tuples. The actual matches are in candidates[i][0].
    # We extract the first group from each tuple which is the full matched model word.
    model_words = [match[0] for match in candidates if match[0]]
    return model_words + get_brand(text)


def extract_model_words_from_value(text):
    if not isinstance(text, str):
        return []
    # Find all matches for value model words
    # This pattern can match either a purely numeric (with optional decimal)
    # or numeric followed by alphabets.
    matches = value_modelword_pattern.findall(text)
    # Each match is a tuple due to groups in regex, the full matched string is the first element of the tuple
    raw_words = [m[0] for m in matches if m[0]]

    # For those that have trailing alphabets (e.g., "123.45abc"), remove the alphabets at the end.
    cleaned_words = []
    for w in raw_words:
        # If pattern matches the form with alphabets: ^\d+(\.\d+)?[a-zA-Z]+$
        # Remove trailing alphabets:
        cleaned = re.sub(r"[a-zA-Z]+$", "", w)
        cleaned_words.append(cleaned)
    return cleaned_words + get_brand(text)


def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def fill_with_most_common(df, column):
    most_common = df[column].mode()[0] if not df[column].mode().empty else "unknown"
    df[column] = df[column].fillna(most_common)
    return df


def lowercase_and_normalize(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    replace = ['"', "''", "inches", "-inch", "-in", "inch", "in"]
    pattern = re.compile(
        r"(" + "|".join(re.escape(word) for word in replace) + r")", flags=re.IGNORECASE
    )
    text = pattern.sub("inch", text)

    replace = ["-hertzes", "hertzes", "-hertz", "-hz", "hz"]
    pattern = re.compile(
        r"(" + "|".join(re.escape(word) for word in replace) + r")", flags=re.IGNORECASE
    )
    text = pattern.sub("hertz", text)
    return text


def normalize_categorical(text):
    if not isinstance(text, str):
        return text
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text


def find_inches(text):
    pattern = r"\b(\d+(?:\.\d+)?)inch\b"
    return re.findall(pattern, text)

def find_hz(text):
    pattern = r"\b(\d+(?:\.\d+)?)hertz\b"
    return re.findall(pattern, text)


def proccess_df(df):
    df_new = df.copy()
    # Step 2: Select core features and handle missing values
    df_new.rename(
        columns={
            "Aspect Ratio": "aspect_ratio",
            "Brand": "brand",
            "Maximum Resolution": "resolution",
            "Screen Size Class": "screen_size",
            "TV Type": "tv_type",
            "Screen Refresh Rate": "refresh_rate",
        },
        inplace=True,
    )
    selected_keys = [
        "brand",
        "resolution",
        "aspect_ratio",
        "tv_type",
        "V-Chip",
        "screen_size",
        "refresh_rate",
    ]

    # Step 4: Handle missing values
    # Fill missing numerical values with the mean and categorical values with the most common value
    numeric_features = ["screen_size", "refresh_rate"]
    categorical_features = ["brand", "resolution", "tv_type", "aspect_ratio", "V-Chip"]

    for feature in numeric_features:  # Convert to numeric
        if feature in df_new.columns:
            # Convert to string, extract numbers, and coerce to numeric
            df_new[feature] = df_new[feature].astype(str).str.extract(r"(\d+)", expand=False)
            df_new[feature] = pd.to_numeric(df_new[feature], errors="coerce", downcast="integer")
        else:
            print(f"Warning: Column '{feature}' is missing in the DataFrame!")

    df_new["title"] = df_new["title"].apply(remove_stopwords)
    df_new["title"] = df_new["title"].apply(lowercase_and_normalize)

    for feature in categorical_features:
        fill_with_most_common(df_new, feature)

    # Extend normalization to all relevant columns
    for column in categorical_features:
        if column in df_new.columns:
            df_new[column] = df_new[column].apply(normalize_categorical)

    df_new["title"] = (
        df_new["title"].apply(extract_model_words_from_title).apply(lambda x: " ".join(x))
    )

    for idx, row in df_new.iterrows():
        if pd.isna(row["screen_size"]):
            inches = find_inches(row["title"])
            if inches:
                df_new.at[idx, "screen_size"] = int(float(inches[0]))
    
    for idx, row in df_new.iterrows():
        if pd.isna(row["refresh_rate"]):
            hertz = find_hz(row["title"])
            if hertz:
                df_new.at[idx, "refresh_rate"] = int(float(hertz[0]))

    for feature in numeric_features:
        fill_with_most_common(df_new, feature)
        #df[feature] = df[feature].apply(lambda x: int(float(x)) if x != "unknown" else x)

    # Fill missing numeric values with mode (since there are -almost- categorical)
    return df_new[["title"] + selected_keys]


# Evaluation Metrics
def evaluate_metrics(candidate_pairs, true_duplicates, total_comparisons):
    """Evaluate metrics for LSH and MSM."""
    duplicates_found = len(set(candidate_pairs).intersection(true_duplicates))

    # Pair Quality (PQ)
    pair_quality = duplicates_found / len(candidate_pairs) if len(candidate_pairs) > 0 else 0

    # Pair Completeness (PC)
    pair_completeness = duplicates_found / len(true_duplicates) if len(true_duplicates) > 0 else 0

    # F1 Measure
    f1 = (
        (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)
        if (pair_quality + pair_completeness) > 0
        else 0
    )

    # Fraction of Comparisons
    fraction_comparisons = len(candidate_pairs) / total_comparisons

    return pair_quality, pair_completeness, f1, fraction_comparisons


def get_duplicates(df):
    model_to_indices = defaultdict(list)
    cnt = 0
    for _, row in df.iterrows():
        model_id = row["modelID"]
        if model_id:
            model_to_indices[model_id].append(cnt)
        cnt += 1

    # Create a set of true duplicates (pairs)
    true_duplicates = set()
    for _, indices in model_to_indices.items():
        if len(indices) > 1:
            # generate all pairs
            for i in indices:
                for j in indices:
                    if i < j:
                        true_duplicates.add((i, j))

    return true_duplicates
