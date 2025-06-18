# Simplified feature_extraction.py for training (updated to support best_box format)
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

POSE_DIR = Path("data/processed/pose_data")
BOX_DIR = Path("data/processed/objects")
LABELS_FILE = Path("data/processed/labels.json")
RESOLUTIONS_FILE = Path("data/processed/video_resolutions.json")
OUTPUT_FILE = Path("data/features/features.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


PROCESSED_FPS = 10

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def calculate_distance(p1: Optional[List], p2: Optional[List]) -> Optional[float]:
    if not p1 or not p2 or None in p1 or None in p2:
        return None
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_velocity(current: Optional[List], previous: Optional[List]) -> Optional[float]:
    return calculate_distance(current, previous) if current and previous else None

def extract_features(video_name: str, label: dict) -> pd.DataFrame:
    pose_path = POSE_DIR / f"{video_name}_pose.json"
    box_path = BOX_DIR / f"detections_{video_name}.json"
    pose_data = load_json(pose_path).get("frames", [])
    box_data = load_json(box_path)

    resolutions = load_json(RESOLUTIONS_FILE)
    frame_width, frame_height = resolutions.get(video_name, (1280, 720))

    action_start_frame = int(label["action_start"]["video_second"] * PROCESSED_FPS)
    action_end_frame = int(label["action_end"]["video_second"] * PROCESSED_FPS)

    rows = []
    prev_box_center = None
    total_frames = max(len(pose_data), len(box_data))

    for i in range(total_frames):
        row = {
            "video_name": video_name,
            "frame_index": i,
            "timestamp": i / PROCESSED_FPS,
            "is_moving": 1 if action_start_frame <= i <= action_end_frame else 0
        }

        # Hand positions
        lw = pose_data[i].get("left_wrist") if i < len(pose_data) else [None, None]
        rw = pose_data[i].get("right_wrist") if i < len(pose_data) else [None, None]
        row.update({
            "left_wrist_x": lw[0], "left_wrist_y": lw[1],
            "right_wrist_x": rw[0], "right_wrist_y": rw[1]
        })

        # Box detection from best_box
        box_frame = box_data[i] if i < len(box_data) else {}
        best_box = box_frame.get("best_box")
        if best_box and best_box.get("confidence", 0) >= 0.5:
            bbox = best_box.get("bbox")
            center = best_box.get("center")
        else:
            bbox = center = None

        if center:
            cx, cy = center
            cx = cx / frame_width
            cy = cy / frame_height
        else:
            cx = cy = None

        row.update({"box_center_x": cx, "box_center_y": cy})

        # Features
        box_velocity = calculate_velocity([cx, cy], prev_box_center)
        lw_dist = calculate_distance(lw, [cx, cy]) if cx is not None else None
        rw_dist = calculate_distance(rw, [cx, cy]) if cx is not None else None
        if lw_dist is not None and rw_dist is not None:
            avg_hand_box_dist = np.mean([lw_dist, rw_dist])
        elif lw_dist is not None:
            avg_hand_box_dist = lw_dist
        elif rw_dist is not None:
            avg_hand_box_dist = rw_dist
        else:
            avg_hand_box_dist = None
        wrist_dist = calculate_distance(lw, rw)

        row.update({
            "box_velocity": box_velocity,
            "avg_hand_box_distance": avg_hand_box_dist,
            "wrist_distance": wrist_dist
        })

        prev_box_center = [cx, cy] if cx is not None else None
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    labels = load_json(LABELS_FILE)
    all_dfs = []

    for entry in labels:
        video_name = Path(entry["video"]).stem
        df = extract_features(video_name, entry)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Simplified feature dataset saved to {OUTPUT_FILE} with shape {final_df.shape}")

if __name__ == "__main__":
    main()
