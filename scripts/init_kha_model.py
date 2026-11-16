import torch
from transformers import MBart50TokenizerFast, AutoModelForSeq2SeqLM

MODEL_PATH = "/home/others/15407/hf_cache/models--facebook--mbart-large-50/snapshots/4ef55a20b36c6903b832e38f0604ab4bdf22c7d6"
SAVE_PATH  = "/home/others/15407/khasi_mt/mbart-kha"

tokenizer = MBart50TokenizerFast.from_pretrained(SAVE_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.resize_token_embeddings(len(tokenizer))

kha_id = tokenizer.convert_tokens_to_ids("kha_IN")
en_id  = tokenizer.convert_tokens_to_ids("en_XX")
print(f"kha_IN id: {kha_id}, en_XX id: {en_id}")

with torch.no_grad():
    model.model.shared.weight[kha_id] = model.model.shared.weight[en_id].clone()
    model.lm_head.weight[kha_id] = model.lm_head.weight[en_id].clone()

model.config.vocab_size = len(tokenizer)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Model + tokenizer saved to", SAVE_PATH)
print("Final vocab size:", model.config.vocab_size)
