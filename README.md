Helmet Checker — Real‑Time Helmet Detection (YOLOv8)

Detect whether people are wearing a helmet or not in images, videos, and live webcam.
This repo blueprint includes data prep (VOC → YOLO/COCO/HDF5), training (5‑epoch quickstart), evaluation & plots, and a live camera app with an optional violation beep.

All paths/instructions below are written for Windows as per your setup.
Root project folder used here:
C:\Users\sagni\Downloads\Helmet Checker

✨ Features

Dataset prep from Pascal‑VOC (Kaggle: Helmet Detection — AndrewMVD) → YOLO labels + COCO JSON + HDF5 + metadata

One‑command train (YOLOv8n, 5 epochs quick baseline)

Automatic evaluation: results.csv, accuracy curves, confusion matrix (robust threshold sweep)

Advanced analysis: per‑class PR curves, AP@50 bars, qualitative GT vs Pred visuals

Live webcam app with on‑screen labels and optional beep for no_helmet

📦 Directory Layout
Helmet Checker/
├─ archive/                            # (your raw Kaggle download)
│  ├─ images/                          # Pascal VOC images
│  └─ annotations/                     # Pascal VOC XMLs
├─ data/
│  ├─ images/{train,val,test}/
│  └─ labels/{train,val,test}/         # YOLO txt
├─ coco/
│  ├─ coco_annotations_train.json
│  ├─ coco_annotations_val.json
│  └─ coco_annotations_test.json
├─ runs/
│  └─ detect/helmet_v8n_5ep.../        # YOLO training outputs (auto)
├─ metrics/
│  ├─ accuracy_curves.png
│  ├─ confusion_matrix.png
│  ├─ no_matches_debug.txt             # only if needed
│  ├─ pr/pr_curves_ap50.png
│  ├─ pr/ap50_bars.png
│  └─ vis/*.jpg                        # GT vs Pred snapshots
├─ captures/                           # frames saved from live app
├─ helmet_dataset.yaml                 # Ultralytics dataset file
├─ helmet_dataset.h5                   # HDF5 packed dataset
├─ helmet_metadata.pkl                 # metadata (classes, counts, splits)
├─ prepare_helmet_dataset.py           # DATA: split & convert (VOC → YOLO/COCO/H5)
├─ train_eval_helmet_5ep_fix.py        # TRAIN + EVAL + PLOTS (robust)
├─ extras_analysis_viz.py              # PR curves, AP bars, GT/Pred visuals
└─ live_helmet_cam.py                  # Live webcam helmet checker

🧰 Requirements

Python 3.10+ recommended

(Optional) NVIDIA GPU + CUDA for faster training/inference

Install dependencies:

pip install ultralytics opencv-python matplotlib pandas scikit-learn pyyaml pillow tqdm h5py lxml --upgrade


If you see PyTorch/CUDA messages, follow Ultralytics/PyTorch guidance to match CUDA version. CPU works as well (slower).

📥 Dataset

Kaggle (recommended small starter): “Helmet Detection — AndrewMVD” (≈764 images, Pascal VOC).
Download and unzip so you have:

Images: C:\Users\sagni\Downloads\Helmet Checker\archive\images

Annotations (XML): C:\Users\sagni\Downloads\Helmet Checker\archive\annotations

This dataset typically uses classes helmet and head (we map head → no_helmet).

1) 🚦 Data Prep (Split + Convert)

Runs a full pipeline:

Split 70/15/15

Convert VOC → YOLO (txt)

Generate COCO JSON per split

Save YAML (for Ultralytics), PKL metadata, HDF5 pack

Script: prepare_helmet_dataset.py
Run:

cd "C:\Users\sagni\Downloads\Helmet Checker"
python prepare_helmet_dataset.py


What you get:

data/images/{train,val,test} + data/labels/{train,val,test}

coco/coco_annotations_*.json

helmet_dataset.yaml

helmet_metadata.pkl, helmet_dataset.h5

Adjusting label names: If your XML uses a variant (e.g., no-helmet), open the script and add to:

VOC_TO_CANON = {"helmet": "helmet", "head": "no_helmet", "no-helmet": "no_helmet"}

2) 🏃 Train + Evaluate (5 Epochs, Quick Start)

Script: train_eval_helmet_5ep_fix.py

Trains YOLOv8n for 5 epochs

Validates best weights

Creates/recovers results.csv

Plots accuracy_curves.png and a robust confusion_matrix.png (auto‑sweeps thresholds if needed)

Runs a sanity check on your val labels and writes debug file if matches are not found

Run:

cd "C:\Users\sagni\Downloads\Helmet Checker"
python train_eval_helmet_5ep_fix.py


Outputs:

runs/detect/helmet_v8n_5ep.../weights/best.pt

runs/detect/helmet_v8n_5ep.../results.csv

metrics/accuracy_curves.png

metrics/confusion_matrix.png

(If needed) metrics/no_matches_debug.txt

Want a slightly stronger baseline? Change BASE_MODEL = "yolov8s.pt" inside the script.

3) 📊 Extra Analysis & Visuals

Script: extras_analysis_viz.py

Per‑class PR curves (@IoU=0.50)

AP@50 bars

GT vs Pred side‑by‑side visuals for first 12 val images

Run:

cd "C:\Users\sagni\Downloads\Helmet Checker"
python extras_analysis_viz.py


Outputs:

metrics/pr/pr_curves_ap50.png

metrics/pr/ap50_bars.png

metrics/vis/*.jpg (GT left, Predictions right)

4) 🎥 Live Webcam Helmet Checker

Script: live_helmet_cam.py

Uses your latest best.pt automatically (from runs/detect/helmet_v8n_5ep*)

Shows HELMET (green) and NO_HELMET (red)

Optional beep on no_helmet after a short streak

Run:

cd "C:\Users\sagni\Downloads\Helmet Checker"
python live_helmet_cam.py


Keys:

Q / Esc — quit

S — save current frame to captures/

Tweak:

If your camera isn’t index 0, open the script and set CAM_INDEX = 1 (or 2).

If your model class names are ["helmet","head"], the app already treats head as no_helmet.

🧪 Quick YOLO CLI (optional)

If you prefer raw CLI:

cd "C:\Users\sagni\Downloads\Helmet Checker"
yolo detect train model=yolov8n.pt data=helmet_dataset.yaml imgsz=640 epochs=5 batch=16 workers=8 name=helmet_v8n_5ep
yolo detect val   model=runs/detect/helmet_v8n_5ep/weights/best.pt data=helmet_dataset.yaml

🧩 Configuration

helmet_dataset.yaml (auto‑generated):

path: "C:/Users/sagni/Downloads/Helmet Checker/data"
train: images/train
val: images/val
test: images/test
names: ["helmet", "no_helmet"]


Change image size/batch: In train_eval_helmet_5ep_fix.py

IMG = 640
BATCH = 16
EPOCHS = 5
BASE_MODEL = "yolov8n.pt"  # or "yolov8s.pt"

🛠 Troubleshooting

1) FileNotFoundError: results.csv
Use the provided training script. It will generate a compact CSV if Ultralytics didn’t create one.

2) RuntimeError: No matches were found; adjust thresholds
The 5‑epoch baseline can be weak. The script now sweeps conf (0.05–0.40) and IoU (0.30, 0.50). If still 0 matches:

Check metrics/no_matches_debug.txt (lists GT counts and prediction histograms per sample)

Verify your labels exist in data/labels/val and use class IDs 0..1 only

Train for a few more epochs (e.g., 20), or switch to yolov8s.pt

3) Webcam not opening
Set CAM_INDEX = 1 (or 2). Close other apps using the camera. On Windows, we try cv2.CAP_DSHOW first.

4) Classes not recognized
If your dataset uses head instead of no_helmet, you’re covered. Otherwise, open live_helmet_cam.py and map names in the detection loop.

🔐 Privacy, Safety & Ethics

Blur faces/plates and display signage where recording occurs.

Keep data secure; limit retention; follow local laws & org policy.

Evaluate bias (e.g., headwear types, lighting, demographics). Measure per‑group error rates.

Model outputs should aid safety and awareness; human review recommended for enforcement.
Author
SAGNIK PATRA
