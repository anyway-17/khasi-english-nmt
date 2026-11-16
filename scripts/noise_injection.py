import json
import random

random.seed(42)

INPUT  = "/home/others/15407/khasi_mt/data/train.jsonl"
OUTPUT = "/home/others/15407/khasi_mt/data/train_noise.jsonl"

NOISE_RATE = 0.15  # fraction of tokens affected per sentence

with open(INPUT) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(pairs)} original pairs")

# Build vocabulary for random replacement
vocab = set()
for p in pairs:
    vocab.update(p["translation"]["kha"].split())
vocab = list(vocab)
print(f"Khasi vocabulary size: {len(vocab)}")

def token_dropout(tokens, rate=NOISE_RATE):
    return [t for t in tokens if random.random() > rate or len(tokens) <= 3]

def token_swap(tokens, rate=NOISE_RATE):
    tokens = tokens.copy()
    n_swaps = max(1, int(len(tokens) * rate / 2))
    for _ in range(n_swaps):
        if len(tokens) < 2:
            break
        i = random.randint(0, len(tokens) - 2)
        tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    return tokens

def token_replace(tokens, rate=NOISE_RATE):
    return [t if random.random() > rate else random.choice(vocab) for t in tokens]

NOISE_FUNCS = [token_dropout, token_swap, token_replace]
NOISE_NAMES = ["dropout", "swap", "replace"]

counts = {"dropout": 0, "swap": 0, "replace": 0}
augmented = []

for pair in pairs:
    kha = pair["translation"]["kha"]
    en  = pair["translation"]["en"]
    tokens = kha.split()

    idx = random.randint(0, 2)
    noise_func = NOISE_FUNCS[idx]
    counts[NOISE_NAMES[idx]] += 1

    noised_tokens = noise_func(tokens)
    noised_kha = " ".join(noised_tokens)

    if len(noised_kha.strip()) == 0:
        noised_kha = kha  # fallback to original

    augmented.append({"translation": {"kha": noised_kha, "en": en}})

combined = pairs + augmented
random.shuffle(combined)

with open(OUTPUT, "w") as f:
    for pair in combined:
        f.write(json.dumps(pair) + "\n")

print(f"\nNoise technique breakdown:")
print(f"  Dropout : {counts['dropout']}")
print(f"  Swap    : {counts['swap']}")
print(f"  Replace : {counts['replace']}")
print(f"\nOriginal pairs  : {len(pairs)}")
print(f"Augmented pairs : {len(augmented)}")
print(f"Total pairs     : {len(combined)}")
print(f"Saved to: {OUTPUT}")
