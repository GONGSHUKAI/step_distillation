MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
MASTER_PORT=${MLP_WORKER_0_PORT:-1235}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
LOGDIR="/cpfs01/gongshukai/step_distillation/logs/ovi_ode_init_0105_overfit"

torchrun \
    --nnodes $NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path configs/ovi_causal_ode_overfit.yaml \
    --logdir $LOGDIR \
    --no_visualize \
    # --disable-wandb
