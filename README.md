# 📦 Object Movement Detection

This project detects **object movement (specifically a box)** from video recordings using a combination of **pose estimation**, **object detection**, and **frame-level feature classification**.

## 🚀 Overview

The pipeline consists of:

- ✅ Data Cleaning
- 📸 Frame Extraction (10 FPS)
- ✋ Hand Pose Estimation (MediaPipe)
- 📦 Box Object Detection (YOLOv8 - fine-tuned)
- 📈 Feature Extraction (pose + object + temporal features)
- 🧠 Model Training (Random Forest / XGBoost)
- 🔍 Evaluation + Confidence Scoring
- 🎥 Inference on New Videos 

---

## 📁 Project Structure

```
ObjectMD/
│
├── data/
│   ├── raw/                    # Original videos (organized in folders by class)
│   ├── processed/
│   │   ├── frames/             # Extracted video frames (10 FPS)
│   │   ├── pose_data/         # Pose estimation JSON outputs
│   │   ├── objects/           # YOLOv8 detection results (per frame)
│   │   └── video_resolutions.json
│   ├── roboflow/              # Labeled dataset used for YOLO training
│   └── labels.json            # Metadata: box size, subject height, start/end times
│
├── ObjectMD/
│   ├── dataset.py
│   ├── pose_estimation.py
│   ├── box_detection.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── Evaluation.py
    └──Visualization.py
│
├── reports/
│   └── figures/               # All training graphs, confusion matrices, etc.
├── visuals/                   # Demo videos and result plots
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone and Create Conda Environment

```bash
git clone https://github.com/yourusername/ObjectMD.git
cd ObjectMD

conda create -n objectmd_ENV python=3.10
conda activate objectmd_ENV
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Also install:
- PyTorch + Torchvision (with MPS or CUDA support)
- MediaPipe
- OpenCV
- scikit-learn
- ultralytics (`pip install ultralytics`)

---

## 🧪 Running the Project

### Step 1: Clean the Dataset

```bash
python -m ObjectMd.dataset
```

### Step 2: Extract Frames (10 FPS)

```bash
python -m ObjectMD.frames
```

### Step 3: Pose Estimation (hands)

```bash
python -m ObjectMD.pose_estimation
```

### Step 4: Box Detection (YOLOv8)

Make sure  `best.pt` is saved in the YOLOv8 directory.

```bash
python -m ObjectMD.features.box_detection
```

### Step 5: Feature Extraction

```bash
python -m ObjectMD.feature_extraction
```

Generates `data/features/features.csv`.

### Step 6: Train Model

```bash
python -m ObjectMD.train
```

Saves model + scaler and plots in `reports/` and `visuals/`.

### Step 7: Run Inference on New Video

```bash
python -m ObjectMD.evaluation 
```
run a test on specific video




---

## 📊 Model Info

- Classifier: `Random Forest` or `XGBoost`
- Input Features:
  - Box center, box speed, hand-to-box distance
  - Smoothed features (rolling window)
  - Confidence-weighted motion
- Output: Frame-wise movement classification (`is_moving = 1/0`)
- Output Format: CSV + optional visualization video

---

## 📈 Results

- Average Accuracy: ~85%
- Most Important Features:
  - `avg_hand_to_box_dist`
  - `box_speed`
- Challenges: inconsistent hand detection, resolution mismatches

---

## 🛠️ Future Work

- Add temporal modeling (LSTM, 1D-CNN)
- Deploy on real-time camera input
- Improve edge-case annotations



---

## 👤 Author

- **Djames (GitHub: @yourusername)**  
- Computer Science & AI Student