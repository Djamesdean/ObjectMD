import json
from pathlib import Path

import pandas as pd

FEATURES_DIR = Path("data/processed/features")
LABEL_FILE = Path("data/processed/labels.json")
FPS = 5  # frames per second

def load_label_data(label_file):
    with open(label_file, "r") as f:
        return json.load(f)

def build_video_label_dict(label_data):
    video_labels = {}
    for entry in label_data:
        video_name = Path(entry["video"]).stem
        start_sec = entry.get("action_start", {}).get("video_second")
        end_sec = entry.get("action_end", {}).get("video_second")
        if start_sec is not None and end_sec is not None:
            start_frame = int(start_sec * FPS)
            end_frame = int(end_sec * FPS)
            video_labels[video_name] = (start_frame, end_frame)
    return video_labels

def label_features_with_movement(features_dir, video_labels):
    for feature_file in features_dir.glob("features_*.csv"):
        video_name = feature_file.stem.replace("features_", "")
        if video_name not in video_labels:
            print(f"Skipping {video_name}: no label info")
            continue
        start_frame, end_frame = video_labels[video_name]
        df = pd.read_csv(feature_file)
        df["is_moving"] = df["frame"].apply(lambda f: 1 if start_frame <= f <= end_frame else 0)
        df.to_csv(feature_file, index=False)
        print(f"Labeled {video_name} with is_moving from frame {start_frame} to {end_frame}")

def main():
    label_data = load_label_data(LABEL_FILE)
    video_labels = build_video_label_dict(label_data)
    label_features_with_movement(FEATURES_DIR, video_labels)
    print("✅ Movement labeling complete.")

if __name__ == "__main__":
    main()