# FILE: i2av_fewstep.sh

# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=1

# 执行批量推理脚本
python ovi_fewstep_batch.py \
    --config_path configs/inference/ovi.yaml \
    --checkpoint_path /cpfs01/gongshukai/logs/distill_ovi_lr2e-6_lr_critic_4e-7_weighted_loss/checkpoint_model_002500/model.pt \
    --csv examples/ti2av_gsk.csv \
    --output_dir outputs/ovi_distill_4step_2500step