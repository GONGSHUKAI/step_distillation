import os
import pandas as pd
import cv2


input_csv = '/cpfs01/gongshukai/step_distillation/data/matrix_audio_ovi.csv'
output_csv = '/cpfs01/gongshukai/step_distillation/data/matrix_audio_ovi_filtered.csv'

# filter out all rows where 'num_frames' < 121, read 'video_path' column and read video, get video size and output to h, w column
def filter_and_update_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df[df['num_frames'] >= 121]  # Filter rows where num_frames < 121

    # Extract video paths and sizes
    video_paths = df['video_path'].tolist()
    sizes = []

    for path in video_paths:
        if os.path.exists(path):
            video = cv2.VideoCapture(path)
            if video.isOpened():
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                sizes.append([str(width), str(height)])
            else:
                sizes.append(['0', '0'])  # Default size if video cannot be opened
            video.release()
        else:
            sizes.append(['0', '0'])  # Default size if file does not exist

    df[['video_width', 'video_height']] = pd.DataFrame(sizes, index=df.index)

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    filter_and_update_csv(input_csv, output_csv)
    print(f"Filtered CSV saved to {output_csv}")