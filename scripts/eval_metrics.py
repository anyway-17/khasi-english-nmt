import sacrebleu

pred_file = "/home/others/15407/khasi_mt/results/noise/generated_predictions.txt"
ref_file  = "/home/others/15407/khasi_mt/results/noise/test_refs.txt"

with open(pred_file) as f:
    preds = [l.strip() for l in f]
with open(ref_file) as f:
    refs = [l.strip() for l in f]

assert len(preds) == len(refs)
print(f"Evaluating {len(preds)} sentences...\n")

bleu = sacrebleu.corpus_bleu(preds, [refs])
chrf = sacrebleu.corpus_chrf(preds, [refs])
ter  = sacrebleu.corpus_ter(preds, [refs])

print(f"BLEU : {bleu.score:.4f}")
print(f"chrF : {chrf.score:.4f}")
print(f"TER  : {ter.score:.4f}")
