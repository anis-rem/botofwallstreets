import pandas as pd
import re

def detect_assets_in_text(text, csv_file_path):
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

            phrases = [p.strip().upper() for p in phrases_raw.split('/')]

            if any(re.search(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE) for phrase in phrases) or \
                    re.search(rf'\b{re.escape(ticker)}\b', text, re.IGNORECASE):
                detected.add(ticker)

    except Exception as e:
        print(f"Error processing CSV file '{csv_file_path}': {e}")

    return list(detected)

def update_detected_assets_column(input_csv_path, text_column, output_column, reference_csv_path):
    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input file.")

    df[output_column] = ''

    for idx, row in df.iterrows():
        text_content = str(row[text_column])
        detected_tickers = detect_assets_in_text(text_content, reference_csv_path)
        df.at[idx, output_column] = ', '.join(detected_tickers) if detected_tickers else ''

    df.to_csv(input_csv_path, index=False)
    print(f"Column '{output_column}' updated and saved to '{input_csv_path}'")

update_detected_assets_column(
    input_csv_path="reddit_financial_posts_classified_balanced.csv",
    text_column="cleaned_content",
    output_column="detected_assets",
    reference_csv_path="matched_rows.csv"
)