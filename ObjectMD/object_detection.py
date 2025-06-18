import json
from pathlib import Path

import cv2
from ultralytics import YOLO

INPUT_DIR = Path("data/processed/frames")
BOX_OUT_DIR = Path("data/processed/objects")
DEMO_DIR = Path("data/demo")  # For visualizations
BOX_OUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("runs/detect/train/weights/best.pt")

demo_saved = 0
MAX_DEMO = 5  # Save some demo images

for video_folder in sorted(INPUT_DIR.iterdir()):
    if not video_folder.is_dir():
        continue

    detections = []
    frame_files = sorted(video_folder.glob("*.jpg"))

    for idx, frame_path in enumerate(frame_files):
        results = model(str(frame_path))
        
        frame_detections = {
            "frame": idx, 
            "boxes": [],  # Support multiple boxes
            "best_box": None,
            "best_confidence": 0.0
        }

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0 and box.conf[0] > 0.3:  # Lower threshold for more detections
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    
                    box_data = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2]  # Useful for tracking movement
                    }
                    
                    frame_detections["boxes"].append(box_data)
                    
                    # Keep track of best detection
                    if confidence > frame_detections["best_confidence"]:
                        frame_detections["best_box"] = box_data
                        frame_detections["best_confidence"] = confidence

        detections.append(frame_detections)
        
        # Save demo visualization (first few good detections)
        if demo_saved < MAX_DEMO and frame_detections["best_box"]:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Draw bounding box
                bbox = frame_detections["best_box"]["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence text
                conf = frame_detections["best_confidence"]
                cv2.putText(frame, f"Box: {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Video: {video_folder.name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save demo
                demo_filename = f"box_demo_{demo_saved+1:02d}_{video_folder.name}_frame{idx:04d}.jpg"
                cv2.imwrite(str(DEMO_DIR / demo_filename), frame)
                demo_saved += 1
                print(f"📦 Saved box demo {demo_saved}/5: {demo_filename}")

    # Save detections
    out_file = BOX_OUT_DIR / f"detections_{video_folder.name}.json"
    with open(out_file, 'w') as f:
        json.dump(detections, f, indent=2)

    # Print detection stats
    detected_frames = len([d for d in detections if d["best_box"] is not None])
    total_boxes = sum(len(d["boxes"]) for d in detections)
    print(f"✅ {video_folder.name}: {detected_frames}/{len(detections)} frames with boxes, {total_boxes} total detections")

print(f"\n🎉 Object detection complete! {demo_saved} demo frames saved")