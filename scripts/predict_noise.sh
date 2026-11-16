#!/bin/bash
#SBATCH --job-name=predict_noise
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/predict_noise_%j.log
#SBATCH --error=/home/others/15407/khasi_mt/logs/predict_noise_%j.err

export PATH=/home/others/15407/.conda/envs/anyway_mt/bin:$PATH

python /home/others/15407/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /home/others/15407/khasi_mt/checkpoints/noise \
    --tokenizer_name /home/others/15407/khasi_mt/mbart-kha/ \
    --source_lang kha_IN \
    --target_lang en_XX \
    --forced_bos_token en_XX \
    --validation_file /home/others/15407/khasi_mt/data/val.jsonl \
    --test_file /home/others/15407/khasi_mt/data/test.jsonl \
    --output_dir /home/others/15407/khasi_mt/results/noise \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --num_beams 4 \
    --use_fast_tokenizer False \
    --fp16 \
    --per_device_eval_batch_size 8
