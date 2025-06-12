import json
from pathlib import Path
from typing import List

# Configurable paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
LABELS_FILE = RAW_DATA_DIR / "annotations.json"
CLEANED_LABELS_FILE = PROCESSED_DATA_DIR / "labels.json"

# 1. Gather all video filenames from all 6 folders
def collect_all_video_files() -> List[str]:
    video_files = []
    for folder in RAW_DATA_DIR.iterdir():
        if folder.is_dir():
            for file in folder.glob("*.mp4"):
                video_files.append(file.name)
    return video_files

# 2. Load the original label file
def load_labels():
    with open(LABELS_FILE, "r") as f:
        return json.load(f)

# 3. Filter labels whose video files exist
def filter_valid_labels(labels, valid_video_names):
    valid_labels = []
    missing_videos = []

    for entry in labels:
        video_name = entry["video"]
        if video_name in valid_video_names:
            valid_labels.append(entry)
        else:
            missing_videos.append(video_name)

    return valid_labels, missing_videos

# 4. Save the cleaned labels
def save_cleaned_labels(labels):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLEANED_LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=4)

# 5. Main function
def main():
    print("🔍 Scanning for video files...")
    valid_videos = collect_all_video_files()
    print(f"✅ Found {len(valid_videos)} videos.")

    print("📖 Loading labels...")
    raw_labels = load_labels()
    print(f"🧾 Loaded {len(raw_labels)} label entries.")

    print("🔎 Matching labels with video files...")
    valid_labels, missing = filter_valid_labels(raw_labels, valid_videos)
    print(f"✅ {len(valid_labels)} valid labels.")
    print(f"⚠️ {len(missing)} labels were dropped due to missing video files.")

    print("💾 Saving cleaned labels...")
    save_cleaned_labels(valid_labels)
    print(f"📁 Cleaned labels saved to: {CLEANED_LABELS_FILE}")

    # Optional: print missing video names
    if missing:
        print("\n🛑 Missing video files:")
        for m in missing:
            print(f" - {m}")

if __name__ == "__main__":
    main()