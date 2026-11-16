#!/bin/bash
#SBATCH --job-name=khasi_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpupart_v100
#SBATCH --time=24:00:00
#SBATCH --output=/home/others/15407/khasi_mt/logs/baseline_%j.log
#SBATCH --error=/home/others/15407/khasi_mt/logs/baseline_err_%j.log

export PATH="/home/others/15407/.conda/envs/anyway_mt/bin:$PATH"

python /home/others/15407/transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /home/others/15407/khasi_mt/mbart-kha/ \
    --source_lang en_XX \
    --target_lang kha_IN \
    --forced_bos_token kha_IN \
    --train_file /home/others/15407/khasi_mt/data/train.jsonl \
    --validation_file /home/others/15407/khasi_mt/data/val.jsonl \
    --output_dir /home/others/15407/khasi_mt/checkpoints/baseline \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --num_beams 4 \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --num_train_epochs 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu \
    --greater_is_better True \
    --gradient_checkpointing \
    --fp16 \
    --report_to none \
    --logging_steps 100 \
    --use_fast_tokenizer False
