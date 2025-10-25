from pipeline import Wan22FewstepInferencePipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import argparse
import torch
import os
import torchvision.transforms.functional as TF
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--h", type=int, default=704)
parser.add_argument("--w", type=int, default=1280)
parser.add_argument("--num_frames", type=int, default=121)
args = parser.parse_args()
assert args.num_frames % 4 == 1, "num_frames must be 1 more than a multiple of 4"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

pipe = Wan22FewstepInferencePipeline(config)
if args.checkpoint_folder is not None:
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "")
        new_key = new_key.replace("_checkpoint_wrapped_module.", "")
        new_key = new_key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    m, u = pipe.generator.load_state_dict(new_state_dict, strict=False)
    assert len(u) == 0, f"Unexpected keys in state_dict: {u}"
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

if args.image is not None:
    img = Image.open(args.image).convert("RGB")
    img = img.resize((args.w, args.h), Image.LANCZOS)
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
    wan22_image_latent = pipe.vae.encode_to_latent(img.unsqueeze(0))
else:
    wan22_image_latent = None
video = (
    pipe.inference(
        noise=torch.randn(
            1,
            (args.num_frames - 1) // 4 + 1,
            48,
            args.h // 16,
            args.w // 16,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        text_prompts=[args.prompt],
        wan22_image_latent=wan22_image_latent,
    )[0]
    .permute(0, 2, 3, 1)
    .cpu()
    .numpy()
)

export_to_video(video, args.output_path, fps=24)
