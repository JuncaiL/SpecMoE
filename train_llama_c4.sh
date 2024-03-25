#!/usr/bin/env bash

#torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
# --model_name_or_path JuncaiL/llama-265m \

echo "using 0-9 data shards amont 1024 data shards in C4 datasets"

python3 -u train_llama.py \
    --model_name_or_path './LLaMA_MoE/' \
    --tokenizer_name huggyllama/llama-7b \
    --dataset_name allenai/c4 \
    --dataset_config_name en \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --cache_dir <cache_dir> \
    --output_dir ./saved_models_c4/ \
    --pretrained_checkpoint <checkpoint_dir>/pytorch_model.bin  \
    --data_files en/c4-train.0000*-of-01024.json.gz \
    --logging_steps 100 \
    $@