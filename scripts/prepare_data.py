import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# Load the xlsx file
data_path = os.path.expanduser("~/khasi_mt/data/English-Khasi Training Data 2025.xlsx")
df = pd.read_excel(data_path, header=None)
df.columns = ["en", "kha"]

# Drop empty rows
df = df.dropna()
df["en"] = df["en"].astype(str).str.strip()
df["kha"] = df["kha"].astype(str).str.strip()
df = df[(df["en"] != "") & (df["kha"] != "")]

print(f"Total sentence pairs: {len(df)}")

# 80/10/10 split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Save as JSONL
output_dir = os.path.expanduser("~/khasi_mt/data/")

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    output_path = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in split_df.iterrows():
            record = {"translation": {"en": row["en"], "kha": row["kha"]}}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {output_path}")

print("Done!")
