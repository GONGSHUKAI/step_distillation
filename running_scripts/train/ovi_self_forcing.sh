MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
MASTER_PORT=${MLP_WORKER_0_PORT:-12350}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
LOGDIR="/root/weights/ovi_self_forcing"

torchrun \
    --nnodes $NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path configs/ovi_self_forcing_dmd_20260101.yaml \
    --logdir $LOGDIR \
    --no_visualize \
    # --disable-wandb 
    # 2>&1 > debug.log
