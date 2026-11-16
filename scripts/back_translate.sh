#!/bin/bash
#SBATCH --job-name=back_translate
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/back_translate_%j.log
#SBATCH --error=/home/others/15407/khasi_mt/logs/back_translate_%j.err

export PATH=/home/others/15407/.conda/envs/anyway_mt/bin:$PATH

INPUT=/home/others/15407/khasi_mt/data/mono/mixed_dataset_cleaned.txt
OUTPUT_DIR=/home/others/15407/khasi_mt/data/mono/bt_output
mkdir -p $OUTPUT_DIR

# Convert plain txt to JSONL for run_translation.py
python3 -c "
import json
with open('$INPUT') as f:
    lines = [l.strip() for l in f if l.strip()]
with open('$OUTPUT_DIR/mono_en.jsonl', 'w') as out:
    for line in lines:
        out.write(json.dumps({'translation': {'en': line, 'kha': ''}}) + '\n')
print(f'Converted {len(lines)} sentences to JSONL')
"

python /home/others/15407/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /home/others/15407/khasi_mt/checkpoints/baseline \
    --tokenizer_name /home/others/15407/khasi_mt/mbart-kha \
    --source_lang en_XX \
    --target_lang kha_IN \
    --forced_bos_token kha_IN \
    --test_file $OUTPUT_DIR/mono_en.jsonl \
    --validation_file /home/others/15407/khasi_mt/data/val.jsonl \
    --output_dir $OUTPUT_DIR \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --num_beams 4 \
    --use_fast_tokenizer False \
    --fp16 \
    --per_device_eval_batch_size 16
