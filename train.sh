mkdir -p ./models/lavt_ln_existBCE_v3
mkdir -p ./checkpoints/

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model lavt \
    --model_id lavt_ln_existBCE_v3 \
    --bert_tokenizer dmis-lab/biobert-base-cased-v1.2 \
    --ck_bert dmis-lab/biobert-base-cased-v1.2 \
    --ln_dataset_root ../../groups/BME/LN_dataset_2D_vlm \
    --batch-size 16 \
    --lr 0.000025 \
    --wd 1e-2 \
    --swin_type base \
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
    --window12 \
    --epochs 50 \
    --img_size 512 \
    --workers 4 \
    --pin_mem \
    --output-dir ./checkpoints/ln_existBCE_v3_weighted \
    --neg_ratio 3 \
    --val_every 5 \
    --early_stop 15 \
    2>&1 | tee ./models/lavt_ln_existBCE_v3/output 2>&1