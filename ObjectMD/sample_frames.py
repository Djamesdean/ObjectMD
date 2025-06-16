from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("data/processed/frames")
TARGET_DIR = Path("data/processed/roboflow_samples")
FRAMES_PER_VIDEO = 1  
TARGET_DIR.mkdir(parents=True, exist_ok=True)

all_video_folders = list(SOURCE_DIR.glob("*"))

counter = 0

for folder in all_video_folders:
    frame_files = list(folder.glob("*.jpg"))

    if len(frame_files) == 0:
        continue

    selected = random.sample(frame_files, min(FRAMES_PER_VIDEO, len(frame_files)))

    for frame_path in selected:
        new_name = f"{folder.name}_{frame_path.name}"
        shutil.copy(frame_path, TARGET_DIR / new_name)
        counter += 1

print(f"✅ Copied {counter} frames from {len(all_video_folders)} videos into {TARGET_DIR}")