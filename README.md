# Crop Pest Object Detection — Architecture Benchmarking

Benchmarks six computer vision architectures for crop pest detection and classification, grouped into three paired experiments: YOLO + ViT, Faster R-CNN + EfficientNet, and EfficientDet + ConvNeXt. Each pair combines one object detector with one image classifier, evaluated independently on the same pest dataset.

## Overview

Crop pest detection is a critical agricultural AI problem. This project systematically evaluates modern detection and classification pipelines on a multi-class pest dataset (bees, wasps, earwigs, moths, and more), including explainability analysis (XAI) for the ViT model.

## Tech Stack

- **Language:** Python 3
- **Frameworks:** PyTorch, Hugging Face Transformers, Ultralytics YOLO, OpenCV
- **Models:**
  - YOLO (`yolo11n`) — object detection
  - Vision Transformer (`vit_base_patch16_224`) — classification + XAI
  - Faster R-CNN (`fasterrcnn_resnet50_fpn`) — object detection
  - EfficientNet (`EfficientNet_B4`) — classification
  - EfficientDet (`tf_efficientdet_d2`) — object detection
  - ConvNeXt (`ConvNeXt-Base`) — classification
- **Environment:** Jupyter Notebook + standalone Python scripts

## Project Structure

```
cv-crop-pest-object-detection-benchmarking/
├── 0_Data_Prep/
│   ├── 0_install_libraries.sh
│   ├── 1_download_data.py
│   ├── 2_generate_classifier_Crop_dataset.py
│   ├── 3_yolo_to_coco.py
│   ├── 4_analyze_class_distribution.py
│   ├── 5_analyze_small_ratio.py / .txt
│   ├── 6_check_anchor_match.py / .txt
│   ├── 7_generate_imbalanced_data.py
│   ├── 8_analyze_insect_feature.py
│   └── 8_insect_feature_stats.csv
├── 1_Yolo_ViT/
│   ├── Yolo/                   # YOLO training and evaluation
│   │   ├── 1_train_Yolo.py
│   │   ├── 2_evaluate_Yolo.py
│   │   ├── Yolo_TrainLog.txt
│   │   └── Yolo_map50.txt
│   └── ViT/                    # ViT training, evaluation, XAI
│       ├── 1_train_ViT.py
│       ├── 2_evaluate_ViT.py
│       ├── ViT_CM.png          # Confusion matrix
│       ├── ViT_TrainLog.txt
│       ├── ViT_eval.txt
│       └── xai_vit_multi/      # XAI attention maps, one per sample class
├── 2_FRCNN_EfficientNet/
│   ├── Faster_RCNN/
│   │   ├── 1_train_faster_RCNN.py
│   │   ├── 2_find_FRCNN_f1_score.py
│   │   ├── 3_evaluate_FRCNN.py
│   │   ├── FRCNN_TrainLog.txt
│   │   ├── FRCNN_map50.txt
│   │   └── FRCNN_P_R_F1.txt
│   └── EfficientNet/
│       ├── 1_train_efficient_net.py
│       ├── 2_evaluate_efficient_net.py
│       ├── EfficientNet_CM.png
│       ├── EfficientNet_TrainLog.txt
│       └── EfficientNet_eval.txt
├── 3_EfficientDet_ConvNext/
│   ├── EfficientDet/
│   │   ├── 1_train_efficient_det.py
│   │   ├── 2_evaluate_efficient_det.py
│   │   ├── NoAugm_efficientDet_TrainLog.txt
│   │   ├── WithAugm_efficientDet_TrainLog.txt
│   │   └── EfficientDet_map50.txt
│   └── ConvNext/
│       ├── 1_train_convNeXt.py
│       ├── 2_evaluate_convNeXt.py
│       ├── ConvNeXt_CM.png
│       ├── ConvNext_TrainLog.txt
│       └── ConvNext_eval.txt
├── Jupyter_version/            # Notebook versions of all four stages
│   ├── (1)Data_prep.ipynb
│   ├── (2)m1_Yolo+ViT.ipynb
│   ├── (3)m2_FRCNN+EfficientNet.ipynb
│   └── (4)m3_EfficientDet+ConvNext.ipynb
└── readme_to_run.txt           # Setup and execution instructions
```

## Pest Classes

Bees, wasps, earwigs, moths (and additional classes from the Roboflow crop pest dataset).

## How to Run

Full setup and execution order (including virtual environment creation) is in `readme_to_run.txt` — replace the hardcoded project path in that file with your own. General pattern:

```bash
# Install dependencies
bash 0_Data_Prep/0_install_libraries.sh
source env/bin/activate

# Step 1: Data prep — download, convert formats, analyze the dataset
python3 0_Data_Prep/1_download_data.py
python3 0_Data_Prep/2_generate_classifier_Crop_dataset.py
python3 0_Data_Prep/3_yolo_to_coco.py
python3 0_Data_Prep/4_analyze_class_distribution.py

# Step 2: Train & evaluate the classifiers
python3 1_Yolo_ViT/ViT/1_train_ViT.py
python3 1_Yolo_ViT/ViT/2_evaluate_ViT.py
python3 2_FRCNN_EfficientNet/EfficientNet/1_train_efficient_net.py
python3 2_FRCNN_EfficientNet/EfficientNet/2_evaluate_efficient_net.py
python3 3_EfficientDet_ConvNext/ConvNext/1_train_convNeXt.py
python3 3_EfficientDet_ConvNext/ConvNext/2_evaluate_convNeXt.py

# Step 3: Train & evaluate the detectors
python3 1_Yolo_ViT/Yolo/1_train_Yolo.py
python3 1_Yolo_ViT/Yolo/2_evaluate_Yolo.py
python3 2_FRCNN_EfficientNet/Faster_RCNN/1_train_faster_RCNN.py
python3 2_FRCNN_EfficientNet/Faster_RCNN/3_evaluate_FRCNN.py
python3 3_EfficientDet_ConvNext/EfficientDet/1_train_efficient_det.py
python3 3_EfficientDet_ConvNext/EfficientDet/2_evaluate_efficient_det.py
```

Or open the notebooks in `Jupyter_version/` directly for an end-to-end walkthrough of each stage.

## Models Compared

| Group | Detector | Classifier |
|-------|----------|------------|
| `1_Yolo_ViT` | YOLO (`yolo11n`) | ViT (`vit_base_patch16_224`) |
| `2_FRCNN_EfficientNet` | Faster R-CNN (`fasterrcnn_resnet50_fpn`) | EfficientNet (`EfficientNet_B4`) |
| `3_EfficientDet_ConvNext` | EfficientDet (`tf_efficientdet_d2`) | ConvNeXt (`ConvNeXt-Base`) |

Each group pairs one object detector with one image classifier, trained and evaluated independently on the same pest dataset. Per-model evaluation artifacts (confusion matrices, mAP@0.5, precision/recall/F1) are produced by each `evaluate_*` script and saved alongside the corresponding training scripts.

## XAI Analysis

Attention maps generated for ViT predictions show which image regions drive classification decisions — useful for validating that the model focuses on pest features rather than background artefacts.
