<p align="center">
<h1 align="center">Wan2.2-TI2V-5B-Turbo</h1>
<a href="https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/quanhaol/MagicData"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

Wan2.2-TI2V-5B-Turbo is designed for efficient step distillation and CFG distillation based on <a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B"><b>Wan2.2-TI2V-5B</b></a>. 

Leveraging the Self-Forcing framework, it enables 4-step TI2V-5B model training. **Our model can generate 121-frame videos at 24 FPS with a resolution of 1280×704 in just 4 steps, eliminating the need for the CFG trick.**

To the best of our knowledge, Wan2.2-TI2V-5B-Turbo is the **first** open-source repository of the distilled I2V version of Wan2.2-TI2V-5B.

## 🔥Video Demos
The videos below can be reproduced using [examples/example.csv](examples/example.csv).

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0b18fa71-fd9c-4991-ae6f-e11555584592" width="100%" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ce12f69f-0945-4626-92b3-276345b3b60b" width="100%" controls loop></video>
      </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/491f4f79-9c01-4ab6-a6ee-9990f3813b70" width="100%" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/2dadcb41-dde0-45c0-85f3-a4be0c3f42d2" width="100%" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/ee4063de-419e-43a5-ab3b-b629c058ba99" width="100%" controls loop></video>
     </td>
  </tr>
</table>

## 📣 Updates
- `2025/08/26` 🔥Wan2.2-TI2V-5B-Turbo has been released [`here`](https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo).

## 🐍 Installation
Create a conda environment and install dependencies:
```bash
conda create -n wanturbo python=3.10 -y
conda activate wanturbo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## 🚀Quick Start

### Checkpoint Download

```bash
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir wan_models/Wan2.2-TI2V-5B
```

### DMD Training 
```bash
bash running_scripts/train/Wan2.2/dmd.sh
```
Our training run uses 4000 iterations and completes in under 2 days using 16 A100 GPUs.

### Fewstep Inference
```bash
bash running_scripts/inference/Wan2.2/i2v_fewstep.sh
```

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects that have been instrumental in the development of our project:

- [CausVid](https://github.com/tianweiy/CausVid)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)

Special thanks to the contributors of these libraries for their hard work and dedication!

## 📚 Contact

If you have any suggestions or find our work helpful, feel free to contact us

Email: liqh24@m.fudan.edu.cn or zhenxingfd@gmail.com or ruiwang16@fudan.edu.cn

If you find our work useful, <b>please consider giving a star to this github repository and citing it</b>:

```bibtex
@article{li2025magicmotion,
  title={MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance},
  author={Li, Quanhao and Xing, Zhen and Wang, Rui and Zhang, Hui and Dai, Qi and Wu, Zuxuan},
  journal={arXiv preprint arXiv:2503.16421},
  year={2025}
}
```
