# 批量模式
CUDA_VISIBLE_DEVICES=7 python wan2.2_fewstep_batch.py \
    --config_path configs/inference/wan22.yaml \
    --checkpoint_folder /videogen/Wan2.2-TI2V-5B-Turbo/logs/distill_anysize/checkpoint_model_003500 \
    --csv /videogen/Wan2.2-TI2V-5B-Turbo/examples/example.csv \