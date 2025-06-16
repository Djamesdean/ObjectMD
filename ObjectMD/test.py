import json
from pathlib import Path

# === CONFIG ===
POSE_DIR = Path("data/processed/pose_data")
BOX_DIR = Path("data/processed/objects")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def debug_data_structure():
    # Get first pose and box files
    pose_files = sorted(POSE_DIR.glob("*_pose.json"))
    
    if not pose_files:
        print("❌ No pose files found!")
        return
    
    pose_file = pose_files[0]
    video_name = pose_file.stem.replace("_pose", "")
    box_file = BOX_DIR / f"detections_{video_name}.json"
    
    print("🔍 Analyzing files:")
    print(f"  Pose file: {pose_file}")
    print(f"  Box file: {box_file}")
    print(f"  Video name: {video_name}")
    print()
    
    # Load and examine pose data
    print("📊 POSE DATA STRUCTURE:")
    pose_data = load_json(pose_file)
    print(f"  Type: {type(pose_data)}")
    
    if isinstance(pose_data, list):
        print(f"  Length: {len(pose_data)}")
        if len(pose_data) > 0:
            print(f"  First item type: {type(pose_data[0])}")
            print(f"  First item keys: {list(pose_data[0].keys())}")
            print("  First item content:")
            for key, value in pose_data[0].items():
                print(f"    {key}: {value}")
            print()
            
            # Check a few more items
            for i in range(min(3, len(pose_data))):
                item = pose_data[i]
                frame_idx = item.get("frame_index", "NO_FRAME_INDEX")
                left_wrist = item.get("left_wrist", "NO_LEFT_WRIST")
                right_wrist = item.get("right_wrist", "NO_RIGHT_WRIST")
                print(f"  Item {i}: frame_index={frame_idx}, left_wrist={left_wrist}, right_wrist={right_wrist}")
                
    elif isinstance(pose_data, dict):
        print(f"  Keys (first 10): {list(pose_data.keys())[:10]}")
        first_key = list(pose_data.keys())[0]
        print(f"  First key: {first_key}")
        print(f"  First value: {pose_data[first_key]}")
    
    print()
    
    # Load and examine box data
    print("📦 BOX DATA STRUCTURE:")
    if box_file.exists():
        box_data = load_json(box_file)
        print(f"  Type: {type(box_data)}")
        print(f"  Length: {len(box_data)}")
        if len(box_data) > 0:
            print(f"  First item: {box_data[0]}")
            print(f"  First item keys: {list(box_data[0].keys())}")
    else:
        print("  ❌ Box file not found!")
    
    print()
    
    # Test the matching logic
    print("🔗 TESTING FRAME MATCHING:")
    if isinstance(pose_data, list) and box_file.exists():
        box_data = load_json(box_file)
        
        # Check first few frames
        for frame_idx in range(min(5, len(box_data))):
            box_frame = box_data[frame_idx]
            
            # Try to find corresponding pose data
            pose_frame = {}
            for pose_entry in pose_data:
                if pose_entry.get("frame_index") == frame_idx:
                    pose_frame = pose_entry
                    break
            
            print(f"  Frame {frame_idx}:")
            print(f"    Box has bbox: {'bbox' in box_frame}")
            print(f"    Pose found: {bool(pose_frame)}")
            if pose_frame:
                print(f"    Left wrist: {pose_frame.get('left_wrist')}")
                print(f"    Right wrist: {pose_frame.get('right_wrist')}")
            print()

if __name__ == "__main__":
    debug_data_structure()