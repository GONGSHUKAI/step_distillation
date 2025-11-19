MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
MASTER_PORT=${MLP_WORKER_0_PORT:-1234}
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
LOGDIR="/videogen/Wan2.2-TI2V-5B-Turbo/logs/distill_ovi_lr2e-6_lr_critic_4e-7_weighted_loss"

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path configs/self_forcing_ovi_dmd.yaml \
    --logdir $LOGDIR \
    --data_path data/matrix_audio_ovi.csv \
    --no_visualize \
    # --disable-wandb