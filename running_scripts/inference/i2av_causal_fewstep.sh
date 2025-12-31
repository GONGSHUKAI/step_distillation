export CUDA_VISIBLE_DEVICES=2

# python ovi_fewstep_causal_batch.py \
#     --config_path configs/inference/self_forcing_ovi.yaml \
#     --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/ovi_original.pt \
#     --csv_path examples/ode_example.csv \
#     --output_dir outputs/ovi_causal \
#     --debug_visual

python ovi_fewstep_causal_batch.py \
    --config_path configs/inference/self_forcing_ovi.yaml \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/outputs/ovi_self_forcing_dmd_20251229/checkpoint_model_002000/checkpoint.pt \
    --csv_path examples/ti2av_gsk.csv \
    --output_dir outputs/ovi_causal \
    --debug_visual

# export CUDA_VISIBLE_DEVICES=1

# python ovi_fewstep_batch.py \
#     --config_path configs/inference/ovi_smallcfg.yaml \
#     --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/ovi_original.pt \
#     --csv examples/ode_example.csv \
#     --output_dir outputs/ovi_causal

# python inference.py \
#     --config_path configs/self_forcing_dmd.yaml \
#     --output_folder outputs/self_forcing_dmd \
#     --checkpoint_path /cpfs01/gongshukai/weights/self-forcing-ode-init/ode_init.pt \
#     --data_path prompts/MovieGenVideoBench_extended.txt  > test.log 2>&1