import json
import random

original  = "/home/others/15407/khasi_mt/data/train.jsonl"
bt_data   = "/home/others/15407/khasi_mt/data/mono/bt_filtered.jsonl"
output    = "/home/others/15407/khasi_mt/data/train_bt_augmented.jsonl"

with open(original) as f:
    orig_pairs = [json.loads(l) for l in f if l.strip()]

with open(bt_data) as f:
    bt_pairs = [json.loads(l) for l in f if l.strip()]

print(f"Original training pairs : {len(orig_pairs)}")
print(f"Back-translated pairs   : {len(bt_pairs)}")

# Mix and shuffle
combined = orig_pairs + bt_pairs
random.seed(42)
random.shuffle(combined)

with open(output, 'w') as out:
    for pair in combined:
        out.write(json.dumps(pair) + "\n")

print(f"Total augmented pairs   : {len(combined)}")
print(f"Saved to: {output}")
