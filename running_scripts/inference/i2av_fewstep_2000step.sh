# FILE: i2av_fewstep.sh

# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=2

# 执行批量推理脚本
python ovi_fewstep_batch.py \
    --config_path configs/inference/ovi.yaml \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/distill_ovi_lr_1e-6_lr_critic_2e-7_weighted_loss/checkpoint_model_002000/model.pt \
    --csv examples/ti2av_gsk.csv \
    --output_dir outputs/ovi_distill_4step_2000step