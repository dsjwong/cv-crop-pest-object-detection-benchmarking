# Crop Pest Object Detection — Architecture Benchmarking

Benchmarks four state-of-the-art computer vision architectures for crop pest detection and classification. Compares YOLO (detection), Vision Transformer (classification), Faster RCNN + EfficientNet, and EfficientDet + ConvNeXt across accuracy, speed, and robustness metrics.

## Overview

Crop pest detection is a critical agricultural AI problem. This project systematically evaluates modern detection and classification pipelines on a multi-class pest dataset (bees, wasps, earwigs, moths, and more), including explainability analysis (XAI) for the ViT model.

## Tech Stack

- **Language:** Python 3
- **Frameworks:** PyTorch, Hugging Face Transformers, Ultralytics YOLO, OpenCV
- **Models:**
  - YOLOv8 (real-time object detection)
  - Vision Transformer (ViT) — classification + XAI
  - Faster RCNN + EfficientNet backbone
  - EfficientDet + ConvNeXt backbone
- **Environment:** Jupyter Notebook + standalone Python scripts

## Project Structure

```
cv-crop-pest-object-detection-benchmarking/
├── 0_Data_Prep/            # Dataset download, split, format conversion
├── 1_Yolo_ViT/
│   ├── YOLO/               # YOLOv8 training and evaluation
│   └── ViT/                # ViT training, evaluation, XAI visualisations
│       ├── 1_train_ViT.py
│       ├── 2_evaluate_ViT.py
│       ├── ViT_CM.png          # Confusion matrix
│       ├── ViT_TrainLog.txt    # Training log
│       ├── ViT_eval.txt        # Evaluation results
│       └── xai_vit_multi/     # XAI attention maps per class
├── 2_FRCNN_EfficientNet/   # Faster RCNN + EfficientNet pipeline
├── 3_EfficientDet_ConvNext/ # EfficientDet + ConvNeXt pipeline
├── Jupyter_version/        # Notebook versions of all experiments
└── readme_to_run.txt       # Setup and execution instructions
```

## Pest Classes

Bees, wasps, earwigs, moths (and additional classes from the Roboflow crop pest dataset).

## How to Run

See `readme_to_run.txt` for environment setup. General pattern:

```bash
pip install torch torchvision ultralytics transformers opencv-python

# Data preparation
python 0_Data_Prep/prepare_dataset.py

# Train & evaluate ViT
python 1_Yolo_ViT/ViT/1_train_ViT.py
python 1_Yolo_ViT/ViT/2_evaluate_ViT.py

# YOLOv8 (via Ultralytics CLI)
yolo detect train data=pest.yaml model=yolov8n.pt epochs=50
```

## Key Results

| Model | Task | Notes |
|-------|------|-------|
| YOLOv8 | Detection | Fast inference, good mAP on small pests |
| ViT | Classification | High accuracy; XAI maps show attention on pest body |
| FRCNN + EfficientNet | Detection | Strong precision, slower inference |
| EfficientDet + ConvNeXt | Detection | Best accuracy / speed trade-off |

## XAI Analysis

Attention maps generated for ViT predictions show which image regions drive classification decisions — useful for validating that the model focuses on pest features rather than background artefacts.
