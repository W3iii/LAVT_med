#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=lavt_one_ln_nnunet_backbone

mkdir -p ./models/${MODEL_ID}
mkdir -p ./checkpoints/

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model lavt_one \
    --backbone nnunet \
    --model_id ${MODEL_ID} \
    --bert_tokenizer ./pretrained_weights/biobert-base-cased-v1.2 \
    --ck_bert ./pretrained_weights/biobert-base-cased-v1.2 \
    --data_root ../../groups/BME/LN_dataset_2D_vlm_npy_2classes_neg10 \
    --batch-size 32 \
    --lr 0.0001 \
    --wd 1e-2 \
    --epochs 100 \
    --img_h 288 \
    --img_w 384 \
    --workers 4 \
    --pin_mem \
    --output-dir ./checkpoints/${MODEL_ID} \
    --neg_ratio 1.0 \
    --fg_fraction 0.333 \
    --batch_dice \
    --n_soft_tokens 6 \
    --seed 42 \
    2>&1 | tee ./models/${MODEL_ID}/output
