import pandas as pd
import re

def detect_assets_in_text(text, csv_file_path):
    """
    Detect tickers mentioned in text based on phrases and tickers from a CSV file.
    Supports multiple phrases per row separated by '/'.

    Args:
        text (str): Input text to scan.
        csv_file_path (str): Path to a CSV file with at least two columns (phrase, ticker).

    Returns:
        list: Unique list of tickers detected in the text.
    """
    detected = set()
    text_upper = text.upper()

    try:
        df = pd.read_csv(csv_file_path)

        if df.shape[1] < 2:
            raise ValueError("CSV file must have at least two columns (phrase, ticker).")

        phrase_col, ticker_col = df.columns[0], df.columns[1]

        for _, row in df.iterrows():
            phrases_raw = str(row[phrase_col])
            ticker = str(row[ticker_col]).upper()

            # Split by '/' and strip whitespace
            phrases = [p.strip().upper() for p in phrases_raw.split('/')]

            # Check if any phrase or the ticker is in the text
            if any(re.search(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE) for phrase in phrases) or \
                    re.search(rf'\b{re.escape(ticker)}\b', text, re.IGNORECASE):
                detected.add(ticker)

    except Exception as e:
        print(f"Error processing CSV file '{csv_file_path}': {e}")

    return list(detected)


def update_detected_assets_column(input_csv_path, text_column, output_column, reference_csv_path):
    """
    Detects tickers in a text column using detect_assets_in_text and writes them into a separate column.
    Overwrites the existing CSV file with the updated values.

    Args:
        input_csv_path (str): Path to the CSV file with text and detection columns.
        text_column (str): Name of the column containing input text.
        output_column (str): Name of the column to overwrite with detected tickers.
        reference_csv_path (str): Path to the phrase–ticker CSV used for detection.

    Returns:
        None
    """
    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in input file.")

    # Initialize the output column with empty strings
    df[output_column] = ''

    # Process each row
    for idx, row in df.iterrows():
        text_content = str(row[text_column])
        detected_tickers = detect_assets_in_text(text_content, reference_csv_path)
        # Join the detected tickers with commas or keep as list representation
        df.at[idx, output_column] = ', '.join(detected_tickers) if detected_tickers else ''

    # Save back to the same file
    df.to_csv(input_csv_path, index=False)
    print(f"✅ Column '{output_column}' updated and saved to '{input_csv_path}'")


# Fixed function call
update_detected_assets_column(
    input_csv_path="reddit_financial_posts.csv",
    text_column="cleaned_content",
    output_column="detected_assets",
    reference_csv_path="matched_rows.csv"
)