#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=lavt_one_ln_v1_augment
CKPT=./checkpoints/${MODEL_ID}/model_best_${MODEL_ID}.pth

CUDA_VISIBLE_DEVICES=0 python test.py \
    --model lavt_one \
    --bert_tokenizer ./pretrained_weights/biobert-base-cased-v1.2 \
    --ck_bert ./pretrained_weights/biobert-base-cased-v1.2 \
    --data_root ../../groups/BME/LN_dataset_2D_vlm_2classes \
    --split test \
    --batch-size 16 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --window12 \
    --img_h 384 \
    --img_w 512 \
    --workers 4 \
    --pin_mem \
    --neg_ratio=-1 \
    --n_soft_tokens 4 \
    --resume ${CKPT} \
    2>&1 | tee ./models/${MODEL_ID}/test_output
