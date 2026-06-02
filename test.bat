@echo off
setlocal

set MODEL_ID=lavt_one_ln_v2_nnunetstyle
set CKPT=.\checkpoints\%MODEL_ID%\model_best_%MODEL_ID%.pth

python test.py ^
    --model lavt_one ^
    --bert_tokenizer .\pretrained_weights\biobert-base-cased-v1.2 ^
    --ck_bert .\pretrained_weights\biobert-base-cased-v1.2 ^
    --data_root ..\DataSet\dataset_2classes_neg10 ^
    --split test ^
    --batch-size 1 ^
    --swin_type base ^
    --pretrained_swin_weights .\pretrained_weights\swin_base_patch4_window12_384_22k.pth ^
    --window12 ^
    --img_h 288 ^
    --img_w 384 ^
    --workers 4 ^
    --neg_ratio=-1 ^
    --n_soft_tokens 6 ^
    --resume %CKPT% ^
    --save_pred ^
    --cc_stats_json .\pred_results\%MODEL_ID%\cc_stats.json ^
    --froc_json .\pred_results\%MODEL_ID%\froc.json ^
    --froc_plot .\pred_results\%MODEL_ID%\froc.png ^
    --size_plot .\pred_results\%MODEL_ID%\size_recall.png ^
    --pred_dir .\pred_results\%MODEL_ID%

endlocal
