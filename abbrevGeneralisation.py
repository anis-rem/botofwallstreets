import pandas as pd
from datasets import load_dataset

def filter_dataset_rows(match_list):
    """
    Read a dataset from Hugging Face and return rows where the second column matches any element in match_list.
    Also returns any elements from match_list that were not found.

    Args:
        match_list (list): List of strings to match against the second column of each row

    Returns:
        tuple: (list of matching row dicts, list of unmatched items)
    """
    matching_rows = []
    unmatched_items = match_list.copy()

    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("rohanmahen/phrase-ticker")
        df = dataset["train"].to_pandas()

        if len(df.columns) < 2:
            print("Error: Dataset must have at least 2 columns")
            return matching_rows, unmatched_items

        second_column = df.columns[1]
        print(f"Comparing against column: '{second_column}'")

        # Filter rows where the second column value is in the match list
        filtered_df = df[df[second_column].isin(match_list)]

        # Remove matched items from unmatched list
        matched_values = set(filtered_df[second_column].unique())
        unmatched_items = [item for item in match_list if item not in matched_values]

        # Convert to list of dictionaries
        matching_rows = filtered_df.to_dict('records')

    except Exception as e:
        print(f"Error loading or processing dataset: {e}")

    return matching_rows, unmatched_items


# Example usage
if __name__ == "__main__":
    strings_to_match = [
        "^GSPC", "^DJI", "^IXIC", "^VIX",
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "JPM", "GS", "BAC", "BRK.B",
        "JNJ", "PG", "KO", "XOM", "PEP", "CVX",
        "BTC", "ETH", "MARA", "RIOT",
        "GLD", "SLV"
    ]

    result, not_found = filter_dataset_rows(strings_to_match)

    print(f"\n✅ Found {len(result)} matching rows:")
    for i, row in enumerate(result, 1):
        print(f"\nRow {i}:")
        for key, value in row.items():
            print(f"  {key}: {value}")

    print(f"\n📦 Total matching rows: {len(result)}")

    if result:
        result_df = pd.DataFrame(result)
        second_column = result_df.columns[1]
        unique_values = result_df[second_column].nunique()
        print(f"\n🔎 Unique values in '{second_column}' column: {unique_values}")
        print(f"Unique values are: {list(result_df[second_column].unique())}")

        # ✅ Save results to CSV
        output_filename = "matched_rows.csv"
        result_df.to_csv(output_filename, index=False)
        print(f"\n💾 Matching rows saved to '{output_filename}'")

    else:
        print("\nNo matching rows found, so no unique values to count or save.")

    if not_found:
        print(f"\n❌ The following {len(not_found)} values were not found in the dataset:")
        print(not_found)
    else:
        print("\n✅ All values from match list were found in the dataset.")
