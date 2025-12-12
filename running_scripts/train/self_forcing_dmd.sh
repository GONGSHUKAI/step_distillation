MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
MASTER_PORT=${MLP_WORKER_0_PORT:-1235}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
LOGDIR="/cpfs01/gongshukai/step_distillation/logs/self_forcing_dmd"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/self_forcing_dmd \
    # --disable-wandb