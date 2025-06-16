import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
POSE_DIR = Path("data/processed/pose_data")
BOX_DIR = Path("data/processed/objects")
LABEL_FILE = Path("data/processed/labels.json")  # optional
OUTPUT_DIR = Path("data/processed/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# === HELPERS ===
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalize_coordinates(x, y):
    """Normalize pixel coordinates to 0-1 range"""
    return x / IMAGE_WIDTH, y / IMAGE_HEIGHT

def normalize_dimensions(w, h):
    """Normalize width and height to 0-1 range"""
    return w / IMAGE_WIDTH, h / IMAGE_HEIGHT

def compute_box_center(bbox):
    """Compute box center and dimensions in pixel space, then normalize"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    # Normalize all values
    cx_norm, cy_norm = normalize_coordinates(cx, cy)
    w_norm, h_norm = normalize_dimensions(w, h)
    
    return cx_norm, cy_norm, w_norm, h_norm

def extract_features(pose_data, box_data):
    features = []
    last_cx, last_cy = None, None
    
    print(f"Processing {len(box_data)} frames")
    
    # Extract frames data from pose_data
    pose_frames = pose_data.get('frames', [])
    print(f"Pose frames available: {len(pose_frames)}")
    
    # Create a lookup dictionary for pose frames by frame_index
    pose_lookup = {}
    for pose_frame in pose_frames:
        frame_idx = pose_frame.get('frame_index')
        if frame_idx is not None:
            pose_lookup[frame_idx] = pose_frame
    
    print(f"Pose lookup created for frames: {sorted(pose_lookup.keys())[:10]}...")  # Show first 10
    
    for box_frame in box_data:
        frame_idx = box_frame.get('frame')
        
        if frame_idx is None:
            print(f"Warning: Box frame missing 'frame' key: {box_frame}")
            continue
        
        # Look for pose data for this frame
        pose_frame = pose_lookup.get(frame_idx, {})
        
        if not box_frame.get("bbox"):
            print(f"Frame {frame_idx}: No bbox found")
            continue

        # Box info - compute normalized center and dimensions
        bbox = box_frame["bbox"]
        cx, cy, w, h = compute_box_center(bbox)

        # Box motion
        if last_cx is not None:
            dx = cx - last_cx
            dy = cy - last_cy
        else:
            dx = dy = 0
        last_cx, last_cy = cx, cy

        # Wrist positions
        lw = pose_frame.get("left_wrist")
        rw = pose_frame.get("right_wrist")

        # Process left wrist
        if lw and len(lw) >= 2 and lw[0] is not None and lw[1] is not None:
            lwx, lwy = lw[0], lw[1]
            # If coordinates seem to be in pixel space (>1), normalize them
            if lwx > 1 or lwy > 1:
                lwx, lwy = normalize_coordinates(lwx, lwy)
        else:
            lwx = lwy = np.nan

        # Process right wrist
        if rw and len(rw) >= 2 and rw[0] is not None and rw[1] is not None:
            rwx, rwy = rw[0], rw[1]
            # If coordinates seem to be in pixel space (>1), normalize them
            if rwx > 1 or rwy > 1:
                rwx, rwy = normalize_coordinates(rwx, rwy)
        else:
            rwx = rwy = np.nan

        # Debug print for first few frames
        if frame_idx < 3:
            print(f"Frame {frame_idx}:")
            print(f"  Box center: ({cx:.3f}, {cy:.3f})")
            print(f"  Pose frame found: {bool(pose_frame)}")
            print(f"  Pose frame keys: {list(pose_frame.keys()) if pose_frame else 'None'}")
            print(f"  Left wrist raw: {lw}")
            print(f"  Right wrist raw: {rw}")
            print(f"  Left wrist processed: ({lwx}, {lwy})")
            print(f"  Right wrist processed: ({rwx}, {rwy})")

        # Distances (now both coordinates are in same normalized space)
        dist_left = euclidean([lwx, lwy], [cx, cy]) if not np.isnan(lwx) else np.nan
        dist_right = euclidean([rwx, rwy], [cx, cy]) if not np.isnan(rwx) else np.nan
        min_dist = np.nanmin([dist_left, dist_right])

        features.append({
            "frame": frame_idx,
            "box_cx": cx,
            "box_cy": cy,
            "box_w": w,
            "box_h": h,
            "box_dx": dx,
            "box_dy": dy,
            "left_wrist_x": lwx,
            "left_wrist_y": lwy,
            "right_wrist_x": rwx,
            "right_wrist_y": rwy,
            "dist_left_to_box": dist_left,
            "dist_right_to_box": dist_right,
            "min_hand_dist": min_dist
        })

    return pd.DataFrame(features)

# === MAIN ===
pose_files = sorted(POSE_DIR.glob("*_pose.json"))

for pose_file in tqdm(pose_files, desc="Extracting features"):
    video_name = pose_file.stem.replace("_pose", "")
    box_file = BOX_DIR / f"detections_{video_name}.json"
    
    if not box_file.exists():
        print(f"Skipping {video_name}: box file not found")
        continue

    print(f"\nProcessing {video_name}")
    
    pose_data = load_json(pose_file)
    box_data = load_json(box_file)
    
    print(f"Pose data structure: {type(pose_data)}")
    print(f"Pose data keys: {list(pose_data.keys())}")
    print(f"Box data length: {len(box_data)}")

    df = extract_features(pose_data, box_data)
    
    # Check if we got any valid data
    if len(df) == 0:
        print(f"Warning: No features extracted for {video_name}")
        continue
    
    # Print summary of extracted features
    print(f"Extracted {len(df)} feature rows")
    print(f"Valid left wrist positions: {df['left_wrist_x'].notna().sum()}")
    print(f"Valid right wrist positions: {df['right_wrist_x'].notna().sum()}")
    
    # Save the features
    output_file = OUTPUT_DIR / f"features_{video_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

print("✅ Feature extraction complete.")