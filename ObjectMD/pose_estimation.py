# src/features/pose_estimation.py

from glob import glob
import json
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Define target output folder
OUTPUT_DIR = Path("data/processed/pose_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_hand_keypoints_from_frame(results):
    
    keypoints = {
        "left_wrist": [None, None],
        "right_wrist": [None, None],
        "confidence": [0.0, 0.0]
    }

    if results.pose_landmarks:
        
        for idx, name in zip([15, 16], ["left_wrist", "right_wrist"]):
            landmark = results.pose_landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                keypoints[name] = [landmark.x, landmark.y]
                keypoints["confidence"][0 if name == "left_wrist" else 1] = landmark.visibility
    return keypoints

def process_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    frame_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = extract_hand_keypoints_from_frame(results)
        frame_data.append({
            "frame_index": frame_idx,
            **keypoints
        })
        frame_idx += 1

    cap.release()
    return frame_data

def main():
    video_paths = glob("data/raw/**/*.mp4", recursive=True)

    print(f"[INFO] Found {len(video_paths)} videos to process.")

    for video_path in tqdm(video_paths):
        video_name = Path(video_path).stem
        out_path = OUTPUT_DIR / f"{video_name}_pose.json"

        if out_path.exists():
            continue  # Skip if already processed

        frame_data = process_video(video_path)

        output_json = {
            "video": Path(video_path).name,
            "frames": frame_data
        }

        with open(out_path, "w") as f:
            json.dump(output_json, f, indent=2)

    print(f"[DONE] Pose estimation data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()