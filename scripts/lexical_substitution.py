import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)

INPUT   = "/home/others/15407/khasi_mt/data/train.jsonl"
OUTPUT  = "/home/others/15407/khasi_mt/data/train_lexsub.jsonl"

# Load original pairs
with open(INPUT) as f:
    pairs = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(pairs)} original pairs")

# Extract all unique Khasi tokens for vocabulary
all_kha_tokens = set()
for p in pairs:
    tokens = p["translation"]["kha"].split()
    all_kha_tokens.update(tokens)

vocab = list(all_kha_tokens)
print(f"Khasi vocabulary size: {len(vocab)}")

# Load LaBSE model
print("Loading LaBSE model...")
model = SentenceTransformer("sentence-transformers/LaBSE")

# Encode entire vocabulary
print("Encoding vocabulary (this may take a few minutes)...")
vocab_embeddings = model.encode(vocab, batch_size=256, show_progress_bar=True)
vocab_embeddings = np.array(vocab_embeddings)

def substitute_tokens(sentence, sub_rate=0.20, top_k=10):
    tokens = sentence.split()
    new_tokens = []
    for token in tokens:
        if random.random() < sub_rate and len(token) > 2:
            # Find token embedding
            token_emb = model.encode([token])
            # Compute similarity with vocab
            sims = cosine_similarity(token_emb, vocab_embeddings)[0]
            # Get top_k similar tokens (excluding itself)
            top_indices = np.argsort(sims)[::-1][1:top_k+1]
            candidates = [vocab[i] for i in top_indices]
            # Pick randomly from top candidates
            new_tokens.append(random.choice(candidates))
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

# Generate 2 augmented copies per sentence
print("Generating augmented pairs...")
augmented = []
for i, pair in enumerate(pairs):
    if i % 2000 == 0:
        print(f"  Processing {i}/{len(pairs)}...")
    kha = pair["translation"]["kha"]
    en  = pair["translation"]["en"]
    for _ in range(2):
        aug_kha = substitute_tokens(kha, sub_rate=0.20, top_k=10)
        augmented.append({"translation": {"kha": aug_kha, "en": en}})

# Combine original + augmented and shuffle
combined = pairs + augmented
random.shuffle(combined)

with open(OUTPUT, "w") as f:
    for pair in combined:
        f.write(json.dumps(pair) + "\n")

print(f"\nOriginal pairs    : {len(pairs)}")
print(f"Augmented pairs   : {len(augmented)}")
print(f"Total pairs       : {len(combined)}")
print(f"Saved to: {OUTPUT}")
