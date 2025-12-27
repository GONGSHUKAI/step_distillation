## Step2: Examine ODE checkpoint
export CUDA_VISIBLE_DEVICES=0

python ovi_fewstep_causal_batch.py \
    --config_path configs/inference/self_forcing_ovi.yaml \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/ovi_ode_init/checkpoint_model_008000/model.pt \
    --csv_path examples/ode_example.csv \
    --output_dir outputs/ovi_causal \
    --debug_visual

# python ovi_fewstep_causal_batch.py \
#     --config_path configs/inference/self_forcing_ovi.yaml \
#     --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/distill_ovi_lr_2e-6_lr_critic_4e-7_weighted_loss_smallcfg_720ckpt_15k_data/checkpoint_model_012000/model.pt \
#     --csv_path examples/ode_example.csv \
#     --output_dir outputs/ovi_causal \
#     --debug_visual