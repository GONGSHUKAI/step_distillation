# python inference.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --output_folder videos/self_forcing_dmd \
#     --checkpoint_path /cpfs01/gongshukai/weights/self-forcing-ode-init/ode_init.pt \
#     --data_path prompts/MovieGenVideoBench_extended.txt \
#     --use_ema

# python inference.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --output_folder videos/self_forcing_dmd \
#     --checkpoint_path /root/weights/Wan2.1-T2V-1.3B/wan2.1_1.3b.pt \
#     --data_path prompts/MovieGenVideoBench_extended.txt \
#     --use_ema

python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/self_forcing_dmd/checkpoint_model_003000/model.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema