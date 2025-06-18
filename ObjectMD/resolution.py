# extract_video_resolutions.py
import json
from pathlib import Path

import cv2

VIDEO_DIR = Path("data/raw")
OUTPUT_JSON = Path("data/processed/video_resolutions.json")

video_resolutions = {}

# Search all .mp4 files in nested folders
video_files = list(VIDEO_DIR.rglob("*.mp4"))
print(f"Found {len(video_files)} video files")

for video_path in video_files:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open: {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = video_path.stem  # removes .mp4 extension

    video_resolutions[name] = (width, height)
    print(f"✅ {name}: {width}x{height}")

    cap.release()

# Save to JSON
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(video_resolutions, f, indent=2)

print(f"\n✅ Saved all resolutions to {OUTPUT_JSON}")
