from .diffusion import Trainer as DiffusionTrainer
from .gan import Trainer as GANTrainer
from .ode import Trainer as ODETrainer
from .ovi_ode import Trainer as OviODETrainer
from .distillation import Trainer as ScoreDistillationTrainer
from .wan22_distillation import Trainer as Wan22ScoreDistillationTrainer
from .ovi_distillation import Trainer as OviScoreDistillationTrainer
from .ovi_distillation_v2 import Trainer as OviScoreDistillationImageVideoTrainer
from .ovi_distillation_v2_rl import Trainer as OviScoreDistillationImageVideoRLTrainer
from .ovi_distillation_v2_reward import Trainer as OviScoreDistillationImageVideoRewardTrainer
__all__ = [
    "DiffusionTrainer",
    "GANTrainer",
    "ODETrainer",
    "OviODETrainer",
    "ScoreDistillationTrainer",
    "Wan22ScoreDistillationTrainer",
    "OviScoreDistillationTrainer",
    "OviScoreDistillationImageVideoTrainer",
    "OviScoreDistillationImageVideoRLTrainer",
    "OviScoreDistillationImageVideoRewardTrainer",
]
