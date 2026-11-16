from transformers import MBart50TokenizerFast

MODEL_PATH = "/home/others/15407/hf_cache/models--facebook--mbart-large-50/snapshots/4ef55a20b36c6903b832e38f0604ab4bdf22c7d6"
SAVE_PATH  = "/home/others/15407/khasi_mt/mbart-kha"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)

print("Token count before:", len(tokenizer))
print("kha_IN already exists?", "kha_IN" in tokenizer.additional_special_tokens)

tokenizer.add_special_tokens({
    "additional_special_tokens": tokenizer.additional_special_tokens + ["kha_IN"]
})

print("Token count after:", len(tokenizer))

kha_id = tokenizer.convert_tokens_to_ids("kha_IN")
print("kha_IN token ID:", kha_id)
assert kha_id != tokenizer.unk_token_id, "Token was not added properly!"

tokenizer.save_pretrained(SAVE_PATH)
print("Tokenizer saved to", SAVE_PATH)
