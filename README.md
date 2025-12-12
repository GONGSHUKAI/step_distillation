<p align="center">
<h1 align="center">StreamingT2AV</h1>
<a href="https://github.com/GONGSHUKAI/step_distillation"><img src="https://img.shields.io/badge/GitHub-Repository-0066cc.svg" alt="GitHub"></a>
<!-- <a href="https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/quanhaol/MagicData"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a> -->

## 配置环境
```bash
conda create -n mediagen python=3.10 -y
conda activate mediagen
# 建议先安装除了flash-attn之外的所有依赖包，然后再安装flash-attn。
pip install -r requirements.txt
pip install flash-attn==2.7.3 --use-pep517 --no-build-isolation
```

## 下载权重

和Ovi-DMD相关的权重
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 下载和Ovi有关的权重, 会下载下来 Ovi 的权重 (但我们不会用到960*960和960*960_10s的ckpt), MMAudio 和 Wan2.2-TI2V-5B
python ovi_weight_download.py
```

## 调整一些路径
有一些路径我都为了方便起见写成了绝对路径，需要修改一下，以下列出了一些（如果还有哪里没有列出的话可以运行的时候排查一下）

1. 配置文件: `configs/`和`configs/inference`底下的`.yaml`文件
2. 运行脚本：`running_scripts/train`和`running_scripts/inference`底下的各种`.sh`文件
3. 代码文件：`scripts/visualize_ode_pairs.py`, `utils/ovi_wrapper.py`, `utils/wan_wrapper.py`, `utils/ovi_wrapper_inference.py`, `utils/dataset.py`

## Ovi-DMD
### 训练
加载原始的Ovi权重，训练成4步推理的Ovi-DMD模型。

权重保存路径、使用的配置文件、训练用的prompt+参考帧数据集见`.sh`脚本
```bash
cd step_distillation
bash running_scripts/train/ovi_dmd_lr_2e-6_lr_critic_4e-7_smallcfg_720ckpt.sh
```
### 推理
这是对训好的4步Ovi-DMD进行推理的脚本。

推理的保存结果、使用的配置文件、用来生成视频的prompt+参考帧数据集见`.sh`脚本
```bash
cd step_distillation
bash running_scripts/inference/i2av_fewstep_4000step.sh
```

## 其他：复现Self-Forcing相关

如果你想要复现一下Self-Forcing的话，跟着Self-Forcing的readme下载一些诸如Wan2.1-T2V-1.3B，Self-Forcing训好的ODE_init.pt等等。这个codebase也能跑通Wan2.1-T2V-1.3B的Self-Forcing的训练和推理。

### 下载Wan2.1-T2V-1.3B(Student模型、Critic模型)，Wan2.1-T2V-14B(Teacher模型)和Self-Forcing的ODE初始化权重
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /your/path/to/Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /your/path/to/Wan2.1-T2V-14B
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir /your/path/to/ode_init
```

### ODE 预训练
#### ODE Pair的生成
记得修改里面的绝对路径
```bash
export PYTHONPATH=$PYTHONPATH:/cpfs01/gongshukai/step_distillation

# 第一步: 生成ODE pairs
torchrun \
    --nproc_per_node 8 \
    scripts/generate_ode_pairs.py \
    --output_folder /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode \
    --caption_path /cpfs01/gongshukai/CausVid/sample_dataset/mixkit_prompts.txt

# 可视化生成的ODE pairs
PYTHONPATH=. python scripts/visualize_ode_pairs.py 

# 第二步: 把生成的ODE pairs打包成LMDB数据集
# [注]: LMDB (Lightweight Database) 可以高效地存储和检索大规模数据
PYTHONPATH=. python scripts/create_lmdb_iterative.py --data_path /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode --lmdb_path /cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode_lmdb
```

#### ODE预训练
记得修改里面的绝对路径
```bash
bash ode_init/ode_pretraining.sh
```

### Self-Forcing训练
这里你可以使用自己生成的ODE pairs训练出来的权重，也可以用Self-Forcing给的ODE初始化权重，这个可以在配置文件`configs/self_forcing_dmd.yaml`中调整，修改`generator_ckpt`字段即可。记得修改里面的绝对路径。
```bash
bash running_scripts/train/self_forcing_dmd.sh
```
