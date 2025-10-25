MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
MASTER_PORT=${MLP_WORKER_0_PORT:-8888}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
LOGDIR="/videogen/Wan2.2-TI2V-5B-Turbo/logs/distill_anysize"

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path configs/self_forcing_wan22_dmd.yaml \
    --logdir $LOGDIR \
    --data_path data/matrix.csv \
    --no_visualize \
    # --disable-wandb