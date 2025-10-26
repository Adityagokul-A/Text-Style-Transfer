import pandas as pd
import json

def convert_shakespeare_dataset_to_json(parquet_path, output_path="shakespeare.json"):
    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(df.columns)
    # Sanity check columns
    if not {"modern", "shakespearean"}.issubset(df.columns):
        raise ValueError("Dataset must have columns named 'modern' and 'shakespeare'.")

    # Build list of dicts for JSON
    data = []
    for _, row in df.iterrows():
        modern_text = str(row["modern"]).strip()
        shakespeare_text = str(row["shakespearean"]).strip()

        # Skip empty rows
        if not modern_text or not shakespeare_text:
            continue

        data.append({
            "modern_sentence": f"Convert to <Shakespeare> from <modern>: {modern_text}",
            "shakespeare_sentence": shakespeare_text
        })

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(data)} examples to {output_path}")


convert_shakespeare_dataset_to_json("train-00000-of-00001.parquet")