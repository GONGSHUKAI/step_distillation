import argparse
import warnings
import os
import sys
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer, Wan22ScoreDistillationTrainer, OviScoreDistillationTrainer, OviScoreDistillationImageVideoTrainer, OviScoreDistillationImageVideoRLTrainer, OviScoreDistillationImageVideoRewardTrainer
import logging, os
logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s] %(levelname)s: %(message)s"
)

def suppress_noise():
    warnings.filterwarnings("ignore", message=".*weights_only=False.*")
    warnings.filterwarnings("ignore", message=".*weight_norm.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    noisy_modules = [
        "audiobox_aesthetics", 
        "audiobox_aesthetics.infer",
        "transformers",
        "diffusers",
        "torchaudio",
        "accelerate", 
        "peft",
    ]
    
    for module_name in noisy_modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank != 0:
        warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="/videogen/Wan2.2-TI2V-5B-Turbo/logs/distill_ovi", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="/videogen/Wan2.2-TI2V-5B-Turbo", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", default=False, action="store_true", help="Disable wandb logging")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode, no saving or visualization")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.data_path = args.data_path if config.data_path is None else config.data_path
    config.debug = args.debug

    suppress_noise()
    
    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "score_distillation_wan22":
        trainer = Wan22ScoreDistillationTrainer(config)
    elif config.trainer == "score_distillation_ovi":
        trainer = OviScoreDistillationImageVideoTrainer(config)
    elif config.trainer == "score_distillation_ovi_rl":
        trainer = OviScoreDistillationImageVideoRLTrainer(config)
    elif config.trainer == "score_distillation_ovi_reward":
        trainer = OviScoreDistillationImageVideoRewardTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
