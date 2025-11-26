import os
import pandas as pd
import cv2


input_csv = '/cpfs01/gongshukai/Ovi/example_prompts/gpt_examples_10s_i2v.csv'
image_dir = '/cpfs01/gongshukai/step_distillation/examples/image_ovi3'
output_csv = '/cpfs01/gongshukai/step_distillation/data/matrix_audio_ovi_new.csv'

def filter_and_update_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # 将列名从text_prompt,image_path改为prompt,image，再加一列seed
    df = df.rename(columns={"text_prompt": "prompt", "image_path": "image"})
    df["seed"] = 42  # Add a new column 'seed' with a default value
    # 如果发现句子中有`Audio: `这样的串，将其改为<AUDCAP>，如果发现这一迹象才同时末尾加上<ENDAUDCAP>，否则不加
    df["prompt"] = df["prompt"].apply(lambda x: x.replace("Audio: ", "<AUDCAP>") + "<ENDAUDCAP>" if "Audio: " in x else x)
    # 替换image path: 原本长这样example_prompts/pngs_10s/19.png，现在改为`image_dir`/19.png
    df["image"] = df["image"].apply(lambda x: os.path.join(image_dir, os.path.basename(x)))
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    filter_and_update_csv(input_csv, output_csv)
    print(f"Filtered CSV saved to {output_csv}")