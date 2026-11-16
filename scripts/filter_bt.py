import json

input_en  = "/home/others/15407/khasi_mt/data/mono/mixed_dataset_cleaned.txt"
input_kha = "/home/others/15407/khasi_mt/data/mono/bt_output/generated_predictions.txt"
output    = "/home/others/15407/khasi_mt/data/mono/bt_filtered.jsonl"

with open(input_en) as f:
    en_lines = [l.strip() for l in f if l.strip()]

with open(input_kha) as f:
    kha_lines = [l.strip() for l in f if l.strip()]

# Align to shorter length (handles off-by-one)
pairs = list(zip(en_lines, kha_lines))
print(f"Total pairs before filtering: {len(pairs)}")

kept = 0
skipped = 0
with open(output, 'w') as out:
    for en, kha in pairs:
        en_len  = len(en.split())
        kha_len = len(kha.split())

        # Skip empty outputs
        if kha_len == 0:
            skipped += 1
            continue

        # Length ratio filter: 0.4 to 2.5
        ratio = kha_len / en_len
        if 0.4 <= ratio <= 2.5:
            out.write(json.dumps({"translation": {"en": en, "kha": kha}}) + "\n")
            kept += 1
        else:
            skipped += 1

print(f"Kept:    {kept}")
print(f"Skipped: {skipped}")
print(f"Saved to: {output}")
