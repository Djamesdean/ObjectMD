# Updated feature_extraction.py - Simplified with essential features only
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

def calculate_distance_normalized(p1: Optional[List], p2: Optional[List]) -> Optional[float]:
    """Calculate distance in normalized space (0-1) for hand-to-box distances"""
    if not p1 or not p2 or None in p1 or None in p2:
        return None
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_distance_pixels(p1: Optional[List], p2: Optional[List], frame_width: int, frame_height: int) -> Optional[float]:
    """Calculate distance in pixel space for movement calculations"""
    if not p1 or not p2 or None in p1 or None in p2:
        return None
    
    # Convert normalized coordinates to pixel coordinates
    p1_pixels = [p1[0] * frame_width, p1[1] * frame_height]
    p2_pixels = [p2[0] * frame_width, p2[1] * frame_height]
    
    return np.linalg.norm(np.array(p1_pixels) - np.array(p2_pixels))

def calculate_velocity_pixels_per_second(current: Optional[List], previous: Optional[List], frame_width: int, frame_height: int, fps: int = PROCESSED_FPS) -> Optional[float]:
    """Calculate velocity in pixels per second"""
    distance = calculate_distance_pixels(current, previous, frame_width, frame_height)
    return distance * fps if distance is not None else None

def calculate_smoothed_velocity(positions: List, frame_gap: int, frame_width: int, frame_height: int, fps: int = PROCESSED_FPS) -> Optional[float]:
    """Calculate smoothed velocity over multiple frames in pixels/second"""
    if len(positions) < frame_gap + 1:
        return None
    
    current = positions[-1]
    previous = positions[-(frame_gap + 1)]
    
    if not current or not previous or None in current or None in previous:
        return None
    
    # Distance in pixels over multiple frames
    distance = calculate_distance_pixels(current, previous, frame_width, frame_height)
    if distance is None:
        return None
    
    # Velocity accounting for time gap
    time_gap = frame_gap / fps
    return distance / time_gap

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
    prev_box = None
    box_history = []  # for smoothed velocity
    hand_history = {'avg': []}
    total_frames = max(len(pose_data), len(box_data))

    for i in range(total_frames):
        row = {
            "video_name": video_name,
            "frame_index": i,
            "timestamp": i / PROCESSED_FPS,
            "is_moving": 1 if action_start_frame <= i <= action_end_frame else 0
        }

        # Pose Features (already normalized 0-1)
        pose_frame = pose_data[i] if i < len(pose_data) else {}
        lw = pose_frame.get("left_wrist", [None, None])
        rw = pose_frame.get("right_wrist", [None, None])

        avg_hand = [
            np.mean([v for v in [lw[0], rw[0]] if v is not None]) if lw[0] is not None or rw[0] is not None else None,
            np.mean([v for v in [lw[1], rw[1]] if v is not None]) if lw[1] is not None or rw[1] is not None else None,
        ]

        row.update({
            "left_hand_x": lw[0], "left_hand_y": lw[1],
            "right_hand_x": rw[0], "right_hand_y": rw[1],
            "avg_hand_x": avg_hand[0], "avg_hand_y": avg_hand[1]
        })

        # Box Features (normalize to 0-1 for consistency with hand data)
        box_frame = box_data[i] if i < len(box_data) else {}
        best_box = box_frame.get("best_box")
        box_detected = 0
        if best_box and best_box.get("confidence", 0) >= 0.5:
            x1, y1, x2, y2 = best_box.get("bbox", [None, None, None, None])
            cx, cy = best_box.get("center", [None, None])
            # Normalize box coordinates to match hand coordinates (0-1)
            cx_norm = cx / frame_width if cx is not None else None
            cy_norm = cy / frame_height if cy is not None else None
            w = (x2 - x1) / frame_width if x2 and x1 else None
            h = (y2 - y1) / frame_height if y2 and y1 else None
            conf = best_box.get("confidence", None)
            box_detected = 1
        else:
            cx_norm = cy_norm = w = h = conf = None

        row.update({
            "box_center_x": cx_norm, "box_center_y": cy_norm,
            "box_width": w, "box_height": h,
            "box_confidence": conf
        })

        # Update histories
        box_history.append([cx_norm, cy_norm] if cx_norm is not None else None)
        hand_history['avg'].append(avg_hand if avg_hand[0] is not None else None)

        # Spatial Relationships - Only avg_hand_to_box_dist
        a_dist = calculate_distance_normalized(avg_hand, [cx_norm, cy_norm]) if cx_norm is not None else None

        row.update({
            "avg_hand_to_box_dist": a_dist
        })

        # Movement Features - Only essential ones
        box_speed = calculate_velocity_pixels_per_second([cx_norm, cy_norm], prev_box, frame_width, frame_height) if cx_norm is not None and prev_box is not None else None
        
        # Smoothed velocity (3-frame window only)
        smooth_box_speed_3 = calculate_smoothed_velocity(box_history, 3, frame_width, frame_height)
        smooth_hand_speed_3 = calculate_smoothed_velocity(hand_history['avg'], 3, frame_width, frame_height)

        row.update({
            "box_speed": box_speed,  # pixels/second
            "smoothed_box_speed_3": smooth_box_speed_3,  # pixels/second
            "smoothed_hand_speed_3": smooth_hand_speed_3  # pixels/second
        })

        # Movement angle (keeping this as it might be useful)
        if cx_norm is not None and prev_box is not None:
            dx_norm = cx_norm - prev_box[0]
            dy_norm = cy_norm - prev_box[1]
            movement_angle = np.arctan2(dy_norm, dx_norm) * 180 / np.pi if dx_norm != 0 or dy_norm != 0 else None
        else:
            movement_angle = None

        row.update({
            "box_movement_angle": movement_angle
        })

        # Validity Flags - Only essential ones
        row.update({
            "left_hand_valid": int(lw[0] is not None and lw[1] is not None),
            "right_hand_valid": int(rw[0] is not None and rw[1] is not None),
            "box_detected": box_detected
        })

        # Update previous values
        prev_box = [cx_norm, cy_norm] if cx_norm is not None else None
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