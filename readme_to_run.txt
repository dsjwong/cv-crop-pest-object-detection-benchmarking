Project is on root path = /home/thomas/My_Programs
PLEASE REPLACE IT WITH YOUR PROJECT FILEPATH

# Step1: Install libraries + Download Dataset
# (1) Execute 0_install_libraries.sh to install libraries
/home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/0_install_libraries.sh

# Create & LOAD a virtual enviroment for this project
source /home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/env/bin/activate
pip list

# (2) Download dateset from kaggle
python3 /home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/1_download_data.py

# (3) Generate Crop dataset for classifiers training
python3 /home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/2_generate_classifier_Crop_dataset.py

# (4) Convert raw yolo dataset to COCO json for training non-yolo detectors
python3 /home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/3_yolo_to_coco.py

# (5) Analyze_class_distribution
python3 /home/thomas/My_Programs/COMP9517_CV/source/0_Data_Prep/4_analyze_class_distribution.py






# Step2: Train all the classifiers(it is faster than training detector)

# (1.1)Train Vision Transformer = "vit_base_patch16_224"
python3 /home/thomas/My_Programs/COMP9517_CV/source/1_Yolo_ViT/ViT/1_train_ViT.py
# (1.2)Evaluate Vision Transformer = "vit_base_patch16_224"
python3 /home/thomas/My_Programs/COMP9517_CV/source/1_Yolo_ViT/ViT/2_evaluate_ViT.py

# (2.1)Train EfficientNet = EfficientNet_B4
python3 /home/thomas/My_Programs/COMP9517_CV/source/2_FRCNN_EfficientNet/EfficientNet/1_train_efficient_net.py
# (2.2)Evaluate EfficientNet = EfficientNet_B4
python3 /home/thomas/My_Programs/COMP9517_CV/source/2_FRCNN_EfficientNet/EfficientNet/2_evaluate_efficient_net.py

# (3.1)Train ConvNext = ConvNeXt-Base
python3 /home/thomas/My_Programs/COMP9517_CV/source/3_EfficientDet_ConvNext/ConvNext/1_train_convNeXt.py
# (3.2)Evaluate ConvNext = ConvNeXt-Base
python3 /home/thomas/My_Programs/COMP9517_CV/source/3_EfficientDet_ConvNext/ConvNext/2_evaluate_convNeXt.py


# Step3: Train all the detectors
# (1.1)Train yolo = yolo11n
python3 /home/thomas/My_Programs/COMP9517_CV/source/1_Yolo_ViT/Yolo/1_train_Yolo.py
# (1.2)Evaluate yolo = yolo11n
python3 /home/thomas/My_Programs/COMP9517_CV/source/1_Yolo_ViT/Yolo/2_evaluate_Yolo.py


# (2.1)Train Faster_RCNN = fasterrcnn_resnet50_fpn
python3 /home/thomas/My_Programs/COMP9517_CV/source/2_FRCNN_EfficientNet/Faster_RCNN/1_train_faster_RCNN.py
# (2.2)Evaluate Faster_RCNN = fasterrcnn_resnet50_fpn
python3 /home/thomas/My_Programs/COMP9517_CV/source/2_FRCNN_EfficientNet/Faster_RCNN/3_evaluate_FRCNN.py

# (3.1)Train EfficientDet = tf_efficientdet_d2
python3 /home/thomas/My_Programs/COMP9517_CV/source/3_EfficientDet_ConvNext/EfficientDet/1_train_efficient_det.py
# (3.2)Evaluate EfficientDet = tf_efficientdet_d2
python3 /home/thomas/My_Programs/COMP9517_CV/source/3_EfficientDet_ConvNext/EfficientDet/2_evaluate_efficient_det.py