#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=lavt_one_ln_v1_aug_geom_only

mkdir -p ./models/${MODEL_ID}
mkdir -p ./checkpoints/

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model lavt_one \
    --model_id ${MODEL_ID} \
    --bert_tokenizer ./pretrained_weights/biobert-base-cased-v1.2 \
    --ck_bert ./pretrained_weights/biobert-base-cased-v1.2 \
    --data_root ../../groups/BME/LN_dataset_2D_vlm_2classes \
    --batch-size 16 \
    --lr 0.000025 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --window12 \
    --epochs 50 \
    --img_h 512 \
    --img_w 512 \
    --workers 4 \
    --pin_mem \
    --output-dir ./checkpoints/${MODEL_ID} \
    --neg_ratio 0.3 \
    --n_soft_tokens 4 \
    --seed 42 \
    2>&1 | tee ./models/${MODEL_ID}/output
