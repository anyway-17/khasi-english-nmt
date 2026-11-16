#!/bin/bash
#SBATCH --job-name=noise_train
#SBATCH --partition=gpupart_v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/noise_train_%j.log
#SBATCH --error=/home/others/15407/khasi_mt/logs/noise_train_%j.err

export PATH=/home/others/15407/.conda/envs/anyway_mt/bin:$PATH

python /home/others/15407/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /home/others/15407/khasi_mt/mbart-kha/ \
    --tokenizer_name /home/others/15407/khasi_mt/mbart-kha/ \
    --source_lang kha_IN \
    --target_lang en_XX \
    --forced_bos_token en_XX \
    --train_file /home/others/15407/khasi_mt/data/train_noise.jsonl \
    --validation_file /home/others/15407/khasi_mt/data/val.jsonl \
    --output_dir /home/others/15407/khasi_mt/checkpoints/noise \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --fp16 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu \
    --predict_with_generate \
    --num_beams 4 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer False \
    --report_to none
