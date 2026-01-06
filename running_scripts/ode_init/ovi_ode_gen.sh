export PYTHONPATH=$PYTHONPATH:/cpfs01/gongshukai/step_distillation
# python inference.py --config-file ovi/configs/inference/inference_fusion.yaml
torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    /cpfs01/gongshukai/step_distillation/scripts/generate_ovi_ode_pairs.py \
    --config-file /cpfs01/gongshukai/step_distillation/configs/ovi_ode_gen.yaml

# Step2: creating LMDB dataset from the ODE pairs generated before
# [Note]: LMDB (Lightweight Database) is a fast memory-mapped key-value database.
# PYTHONPATH=. python scripts/create_ovi_lmdb_iterative.py --data_path /root/ermu2001/CODES/Ovi/tmp/outputs/ovi_ode_pairs  --lmdb_path /root/ermu2001/CODES/Ovi/tmp/outputs/ovi_ode_pairs_lmdb