import os
import pandas as pd

# 原始CSV路径
input_csv = '/cpfs01/gongshukai/step_distillation/data/matrix_audio_ovi.csv'
output_csv = '/cpfs01/gongshukai/step_distillation/data/matrix_audio_ovi_filtered.csv'

# 新前缀
old_prefix = '/videogen/audio_preprocess/matrix/'
new_prefix = '/cpfs01/gongshukai/datasets/matrix_data/'

# 读取CSV
df = pd.read_csv(input_csv)

# 替换路径前缀
df['video_path'] = df['video_path'].str.replace(old_prefix, new_prefix, regex=False)
df['audio_path'] = df['audio_path'].str.replace(old_prefix, new_prefix, regex=False)

# 检查文件是否存在
def files_exist(row):
    return os.path.isfile(row['video_path']) and os.path.isfile(row['audio_path'])

# 过滤掉不存在的文件
df_filtered = df[df.apply(files_exist, axis=1)]

# 保存新CSV
df_filtered.to_csv(output_csv, index=False)

print(f"原始条目数: {len(df)}")
print(f"过滤后条目数: {len(df_filtered)}")
print(f"保存为: {output_csv}")