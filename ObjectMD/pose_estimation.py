import json
from pathlib import Path

import cv2
import mediapipe as mp

INPUT_DIR = Path("data/processed/frames")
POSE_OUT_DIR = Path("data/processed/pose_data")
POSE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use Pose instead of Hands for better full-body tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,  # Better for video sequences
    model_complexity=1,       # Good balance of speed/accuracy
    enable_segmentation=False,
    min_detection_confidence=0.5,  # Lower threshold
    min_tracking_confidence=0.5
)

for video_folder in sorted(INPUT_DIR.iterdir()):
    if not video_folder.is_dir():
        continue

    pose_data = []
    frame_files = sorted(video_folder.glob("*.jpg"))

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_result = {
            "frame_index": idx, 
            "left_wrist": [None, None], 
            "right_wrist": [None, None],
            "confidence": None
        }

        if results.pose_landmarks:
            # Get wrist landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Left wrist (landmark 15)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            if left_wrist.visibility > 0.5:  # Check visibility
                frame_result["left_wrist"] = [left_wrist.x, left_wrist.y]
            
            # Right wrist (landmark 16)
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            if right_wrist.visibility > 0.5:
                frame_result["right_wrist"] = [right_wrist.x, right_wrist.y]
                
            frame_result["confidence"] = min(left_wrist.visibility, right_wrist.visibility)

            # Visualization
            annotated = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            demo_dir = Path("data/visuals/pose") / video_folder.name
            demo_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(demo_dir / f"frame_{idx:04d}.jpg"), annotated)
        
        pose_data.append(frame_result)

    # Save results
    out_file = POSE_OUT_DIR / f"{video_folder.name}_pose.json"
    with open(out_file, 'w') as f:
        json.dump({"frames": pose_data}, f, indent=2)

    # Print detection stats
    detected_frames = len([f for f in pose_data if f["left_wrist"][0] is not None or f["right_wrist"][0] is not None])
    print(f"✅ {video_folder.name}: {detected_frames}/{len(pose_data)} frames with hand detection")

pose.close()