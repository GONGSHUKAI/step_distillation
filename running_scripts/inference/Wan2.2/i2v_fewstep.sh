CUDA_VISIBLE_DEVICES=0 python wan2.2_fewstep.py --config_path configs/inference/wan22.yaml \
    --checkpoint_folder wan_models/Wan2.2-TI2V-5B-Turbo \
    --seed $SEED \
    --prompt $PROMPT \
    --image $IMAGE \
    --h 704 --w 1280 \
    --output_path $OUTPATH