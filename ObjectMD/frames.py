# src/data/extract_frames.py

import json
from pathlib import Path

import cv2
from tqdm import tqdm

# --- Configuration ---
VIDEOS_DIR = Path("data/raw")
LABELS_PATH = Path("data/processed/labels.json")
OUTPUT_DIR = Path("data/processed/frames")
FPS = 5  # frames per second to extract

def extract_frames_from_video(video_path: Path, output_folder: Path, fps: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Failed to open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if video_fps > fps else 1

    output_folder.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frame_name = f"frame_{saved_idx:04d}.jpg"
            frame_path = output_folder / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {saved_idx} frames from {video_path.name}")

def load_labeled_video_names(label_path: Path):
    with open(label_path, "r") as f:
        labels = json.load(f)
    return {entry["video"] for entry in labels}

def main():
    print("🔍 Starting frame extraction...")
    labeled_videos = load_labeled_video_names(LABELS_PATH)

    for video_file in tqdm(list(VIDEOS_DIR.rglob("*.mp4"))):
        if video_file.name not in labeled_videos:
            print(f"⏭️ Skipping {video_file.name} — not in labels")
            continue

        video_stem = video_file.stem
        output_path = OUTPUT_DIR / video_stem
        extract_frames_from_video(video_file, output_path, FPS)

    print("✅ All done.")

if __name__ == "__main__":
    main()