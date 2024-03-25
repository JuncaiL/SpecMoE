#!/usr/bin/env bash

#torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
# --model_name_or_path JuncaiL/llama-265m  \
python3 -u train_llama.py \
    --model_name_or_path './LLaMA_MoE/' \
    --tokenizer_name huggyllama/llama-7b \
    --dataset_name wikipedia \
    --dataset_config_name 20220301.en \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --cache_dir <cache_dir> \
    --output_dir ./saved_models/ \
    --logging_steps 100 \
    $@