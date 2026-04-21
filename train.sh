mkdir -p ./models/lavt_ln_existBCE_crop
mkdir -p ./checkpoints/

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model lavt \
    --model_id lavt_ln_existBCE_crop \
    --bert_tokenizer dmis-lab/biobert-base-cased-v1.2 \
    --ck_bert dmis-lab/biobert-base-cased-v1.2 \
    --ln_dataset_root ../../groups/BME/LN_dataset_2D_vlm \
    --batch-size 16 \
    --lr 0.00005 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --window12 \
    --epochs 100 \
    --img_size 384 \
    --workers 4 \
    --pin_mem \
    --output-dir ./checkpoints/ln_existBCE_v2_weighted_crop \
    --neg_ratio 1 \
    --patch_size 128 \
    --fg_prob 0.67 \
    --iters_per_epoch 1000 \
    --val_every 10 \
    --early_stop 15 \
    2>&1 | tee ./models/lavt_ln_existBCE_crop/output 2>&1