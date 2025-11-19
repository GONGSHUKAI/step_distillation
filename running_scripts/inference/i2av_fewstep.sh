# FILE: i2av_fewstep.sh

# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=2

# 执行批量推理脚本
python ovi_fewstep_batch.py \
    --config_path configs/inference/ovi.yaml \
    --checkpoint_path /videogen/Wan2.2-TI2V-5B-Turbo/logs/distill_ovi/checkpoint_model_001000/model.pt \
    --csv examples/ti2av_gsk.csv \
    --output_dir outputs/ovi_inference