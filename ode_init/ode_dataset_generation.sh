export PYTHONPATH=$PYTHONPATH:/cpfs01/gongshukai/step_distillation
# Step1: generating ODE pairs
torchrun \
    --nproc_per_node 8 \
    scripts/generate_ode_pairs.py \
    --output_folder /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode \
    --caption_path /cpfs01/gongshukai/CausVid/sample_dataset/mixkit_prompts.txt

# Step2: creating LMDB dataset from the ODE pairs generated before
# [Note]: LMDB (Lightweight Database) is a fast memory-mapped key-value database.
PYTHONPATH=. python scripts/create_lmdb_iterative.py --data_path /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode --lmdb_path /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode_lmdb