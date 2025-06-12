import json
from pathlib import Path
import random

import cv2

# === CONFIG ===
NUM_VIDEOS = 3
FPS = 30  # default fallback
SHOW_INFO = True

# Get paths relative to script location
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
LABELS_FILE = PROCESSED_DATA_DIR / "labels.json"

# === LOAD LABELS ===
with open(LABELS_FILE, "r") as f:
    labels = json.load(f)

# === VISUALIZE N VIDEOS ===
random.shuffle(labels)
for entry in labels[:NUM_VIDEOS]:
    video_file = entry["video"]

    # Search for the video in all subfolders of data/raw
    matches = list(RAW_DATA_DIR.rglob(video_file))
    if not matches:
        print(f"[!] ❌ Label video name not found: {video_file}")
        continue

    # Confirm exact match
    video_path = matches[0]
    actual_file_name = video_path.name
    if actual_file_name != video_file:
        print(f"[!] ⚠️ Name mismatch! Label: {video_file} | Actual file: {actual_file_name}")
    else:
        print(f"[✓] Matched: {video_file}")

    print(f"▶️ Playing {video_file}")
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = current_frame / fps

            # Decide action status
            start = entry["action_start"]["video_second"]
            end = entry["action_end"]["video_second"]

            if timestamp < start:
                status = "Before Action"
                color = (255, 255, 255)
            elif start <= timestamp <= end:
                status = "During Action"
                color = (0, 255, 0)
            else:
                status = "After Action"
                color = (0, 0, 255)

            # Overlay text
            overlay_text = f"{video_file} | Time: {timestamp:.2f}s | {status}"
            cv2.putText(frame, overlay_text, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if SHOW_INFO:
                subject_info = f"{entry['subject_gender']}, {int(entry['subject_age'])}y"
                cv2.putText(frame, subject_info, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Show the frame
            cv2.imshow("Video Annotation Viewer", frame)

        # WaitKey for pause/play
        key = cv2.waitKey(30 if not paused else 0)
        if key == ord(" "):  # Spacebar
            paused = not paused
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()