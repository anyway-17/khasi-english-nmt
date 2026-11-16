#!/bin/bash
#SBATCH --job-name=baseline_predict
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/predict_%j.log

export PATH=/home/others/15407/.conda/envs/anyway_mt/bin:$PATH

python /home/others/15407/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /home/others/15407/khasi_mt/checkpoints/baseline \
    --source_lang en_XX \
    --target_lang kha_IN \
    --forced_bos_token kha_IN \
    --validation_file /home/others/15407/khasi_mt/data/val.jsonl \
    --test_file /home/others/15407/khasi_mt/data/test.jsonl \
    --output_dir /home/others/15407/khasi_mt/results/baseline \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --num_beams 4 \
    --use_fast_tokenizer False \
    --fp16 \
    --per_device_eval_batch_size 8
