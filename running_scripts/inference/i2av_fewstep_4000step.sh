# FILE: i2av_fewstep.sh

# 设置使用的GPU卡号
export CUDA_VISIBLE_DEVICES=7

# 执行批量推理脚本
python ovi_fewstep_batch.py \
    --config_path configs/inference/ovi_smallcfg.yaml \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/distill_ovi_lr_2e-6_lr_critic_4e-7_weighted_loss_smallcfg_720ckpt_15k_data/checkpoint_model_004000/model.pt \
    --csv examples/ti2av_gsk.csv \
    --output_dir outputs/ovi_distill_4step_4000step