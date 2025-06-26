import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load models
pose_model = mp.solutions.pose.Pose(static_image_mode=True)
box_model = YOLO("runs/detect/train/weights/best.pt")
classifier = joblib.load("models/best_box_movement_classifier.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

# Constants
FPS = 10
CONFIDENCE_THRESHOLD = 0.5


def extract_frames(video_path, output_dir, fps=FPS):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // fps)
    count = 0
    saved = 0
    resolution = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_path = output_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            if resolution is None:
                resolution = (frame.shape[1], frame.shape[0])
            saved += 1
        count += 1

    cap.release()
    return resolution, saved


def extract_pose(frame_path):
    image = cv2.imread(str(frame_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if not results.pose_landmarks:
        return None, 0.0

    left = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

    return {
        'left_wrist': (left.x, left.y) if left.visibility > 0.5 else (None, None),
        'right_wrist': (right.x, right.y) if right.visibility > 0.5 else (None, None),
        'pose_confidence': (left.visibility + right.visibility) / 2
    }, image.shape[:2]


def detect_box(frame_path):
    results = box_model(frame_path, verbose=False)[0]
    boxes = results.boxes
    if not boxes or len(boxes.cls) == 0:
        return None

    box = boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    conf = box.conf[0].item()

    if conf < CONFIDENCE_THRESHOLD:
        return None

    return {
        'bbox': [x1, y1, x2, y2],
        'center': [cx, cy],
        'confidence': conf
    }


def extract_features(video_path):
    video_name = video_path.stem
    temp_frame_dir = Path("temp_frames") / video_name
    resolution, frame_count = extract_frames(video_path, temp_frame_dir)
    width, height = resolution

    features = []
    prev_box = None
    for i in range(frame_count):
        frame_path = temp_frame_dir / f"frame_{i:03d}.jpg"

        pose_data, _ = extract_pose(frame_path)
        box_data = detect_box(frame_path)

        if not box_data:
            continue

        cx, cy = box_data['center'][0] / width, box_data['center'][1] / height
        box_w = (box_data['bbox'][2] - box_data['bbox'][0]) / width
        box_h = (box_data['bbox'][3] - box_data['bbox'][1]) / height
        conf = box_data['confidence']

        if prev_box:
            dx = box_data['center'][0] - prev_box['center'][0]
            dy = box_data['center'][1] - prev_box['center'][1]
            speed = np.sqrt(dx**2 + dy**2) * FPS
        else:
            speed = 0
        prev_box = box_data

        lw = pose_data['left_wrist'] if pose_data else (None, None)
        rw = pose_data['right_wrist'] if pose_data else (None, None)

        lw_dist = np.linalg.norm([lw[0] - cx, lw[1] - cy]) if lw[0] else None
        rw_dist = np.linalg.norm([rw[0] - cx, rw[1] - cy]) if rw[0] else None

        if lw_dist and rw_dist:
            avg_hand_dist = (lw_dist + rw_dist) / 2
        else:
            avg_hand_dist = lw_dist or rw_dist or None

        features.append({
            'frame': i,
            'box_center_x': cx,
            'box_center_y': cy,
            'box_width': box_w,
            'box_height': box_h,
            'box_confidence': conf,
            'box_speed': speed,
            'avg_hand_to_box_dist': avg_hand_dist,
            'pose_confidence': pose_data['pose_confidence'] if pose_data else 0.0
        })

    return pd.DataFrame(features)


def run_inference(video_path):
    df = extract_features(video_path)
    df = df.dropna()

    # Feature engineering
    df['smoothed_box_speed_3'] = df['box_speed'].rolling(window=3, min_periods=1).mean()
    df['speed_ratio'] = df['smoothed_box_speed_3'] / (df['box_speed'] + 1e-8)
    df['speed_diff'] = df['box_speed'] - df['smoothed_box_speed_3']
    df['dist_speed_ratio'] = df['avg_hand_to_box_dist'] / (df['box_speed'] + 1e-8)
    df['confidence_speed'] = df['box_confidence'] * df['box_speed']

    if len(df) > 10:
        df['speed_rolling_mean'] = df['box_speed'].rolling(window=5, min_periods=1).mean()
        df['speed_rolling_std'] = df['box_speed'].rolling(window=5, min_periods=1).std().fillna(0)

    # Match training feature set
    
    with open("models/feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    probs = classifier.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.3).astype(int)

    df['is_moving_pred'] = preds
    df['confidence'] = probs

    output_path = Path("results") / f"{video_path.stem}_prediction.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Prediction saved to: {output_path}")

    return df


if __name__ == "__main__":
    input_video = Path("/Users/merkava/Documents/work/ObjectMD/data/raw/video_env_1_oblique_anonymized/gd_0020.mp4")
    result_df = run_inference(input_video)

    # Check the distribution of predictions
    print(result_df['is_moving_pred'].value_counts())
    print("\nConfidence Range:")
    print(result_df['confidence'].describe())

    # Check key feature distributions
    print("\nKey Features:")
    print(result_df[['box_speed', 'box_confidence', 'avg_hand_to_box_dist']].describe())
    print(result_df[['frame', 'is_moving_pred', 'confidence']].head())