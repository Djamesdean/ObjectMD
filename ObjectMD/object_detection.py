from glob import glob
import json
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO  # Make sure ultralytics is installed

CONFIDENCE_THRESHOLD = 0.5

# Path setup
VIDEO_DIR = Path("data/raw/")
OUTPUT_DIR = Path("data/processed/objects/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO("runs/detect/train/weights/best.pt")  

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    video_name = video_path.stem
    detections = []
    
    #Optional for saving demo video with box hgihlighted
    '''fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    demo_out = cv2.VideoWriter(str(OUTPUT_DIR / f"demo_{video_name}.mp4"), fourcc, fps, (width, height))'''

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)[0]  # Only take first result

        for result in results.boxes:
            cls_id = int(result.cls)
            label = model.names[cls_id]

            # Filter detections to include only box or package and confidence threshold
            if label.lower() in ["box", "package"]:
                conf = float(result.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                detections.append({
                    "frame": frame_idx,
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw bounding box and label on the frame
                '''cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label}: {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        demo_out.write(frame)'''

        frame_idx += 1

    cap.release()
    #demo_out.release()
    # Save to JSON
    out_path = OUTPUT_DIR / f"detections_{video_name}.json"
    with open(out_path, "w") as f:
        json.dump(detections, f, indent=2)

def process_all_videos():
    video_paths = glob('data/raw/**/*.mp4', recursive=True)
    for video_path in tqdm(video_paths, desc="Running object detection"):
        detect_objects_in_video(Path(video_path))

if __name__ == "__main__":
    process_all_videos()