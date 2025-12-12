## Step2: Examine ODE checkpoint
python inference.py \
    --config_path configs/wan_causal_ode.yaml \
    --output_folder outputs/ode_init_samples \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/wan_ode_init/checkpoint_model_004000/model.pt \
    --data_path /cpfs01/gongshukai/Reward-Forcing/prompts/MovieGenVideoBench_extended.txt
    
python inference.py \
    --config_path configs/wan_causal_ode.yaml \
    --output_folder outputs/ode_init_samples \
    --checkpoint_path /cpfs01/gongshukai/weights/self-forcing-ode-init/ode_init.pt \
    --data_path /cpfs01/gongshukai/Reward-Forcing/prompts/MovieGenVideoBench_extended.txt
    
## Step3: Examine Self-forcing checkpoint

python inference.py \
    --config_path configs/inference/self_forcing_wan21.yaml \
    --output_folder outputs/self_forcing_dmd \
    --checkpoint_path /cpfs01/gongshukai/step_distillation/logs/self_forcing_dmd/checkpoint_model_004000/model.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema