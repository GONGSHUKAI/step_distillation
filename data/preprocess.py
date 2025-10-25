#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 target_prompts.jsonl + 视频目录
生成 MagicData.csv（仅三列）
text,video_path,num_frames
"""

import json, csv
from pathlib import Path
import cv2  # 仅用来快速读帧数，pip install opencv-python

# ------------- 可修改的常量 -------------
JSONL_FILE = "/videogen/audio_preprocess/matrix/target_prompts.jsonl"
VIDEO_DIR  = "/videogen/audio_preprocess/matrix/video"
OUTPUT_DIR = "/videogen/audio_preprocess/matrix"
CSV_NAME   = "/videogen/Wan2.2-TI2V-5B-Turbo/data/matrix.csv"
MAX_ENTRIES = None
# ---------------------------------------

jsonl_path   = Path(JSONL_FILE)
video_dir    = Path(VIDEO_DIR)
output_dir   = Path(OUTPUT_DIR)
csv_path     = CSV_NAME

# 1. 读 jsonl
prompts = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        long_caption = data.get("ig25f_caption", {}).get("long_caption", "")
        clip_id = data.get("clip_id", "")
        if not clip_id or not long_caption:
            continue
        prompts.append((clip_id, long_caption.strip().replace("\n", " ")))

# 2. 过滤并补帧数
existing_videos = {p.stem: p for p in video_dir.glob("*.mp4")}
rows = []
for cid, cap in prompts:
    if cid not in existing_videos:
        continue
    vpath = existing_videos[cid]
    # 快速读帧数
    cap_cv = cv2.VideoCapture(str(vpath))
    num_frames = int(cap_cv.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_cv.release()
    rows.append((cap, str(vpath), num_frames))
    if MAX_ENTRIES is not None and len(rows) >= MAX_ENTRIES:
        break

# 3. 写 CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "path", "num_frames"])
    w.writerows(rows)

print(f"Done! 共 {len(rows)} 条 → {csv_path}")