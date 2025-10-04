# Project 4: Object Detection with YOLO

## Overview
Comprehensive computer vision project implementing YOLO (You Only Look Once) for object detection. Includes multiple custom datasets and different YOLO versions.

## Project Structure
```
4 - Object Detection - YOLO/
├── Aerial Cars - Custom Dataset/           # Detect cars from aerial imagery
├── FacialMaskDetector - Custom Dataset/    # COVID mask detection
├── Object Detection/                       # General object detection
└── Object Detection YOLOv4/                # YOLOv4 implementation
```

## Learning Objectives
- Computer vision fundamentals
- Object detection algorithms
- YOLO architecture (v4, v5)
- Training custom detectors
- Real-time object detection
- Dataset preparation and annotation

## Custom Datasets

### 1. Aerial Cars Detection
- **Use Case**: Traffic monitoring, parking management
- **Dataset**: Aerial/drone imagery of vehicles
- **Application**: Urban planning, traffic analysis

### 2. Facial Mask Detector
- **Use Case**: COVID-19 safety compliance
- **Dataset**: People wearing/not wearing masks
- **Application**: Public health monitoring

### 3. General Object Detection
- **Use Case**: Multi-class object recognition
- **Dataset**: Various everyday objects
- **Application**: General-purpose detection

## YOLO Versions Implemented
- **YOLOv4**: High accuracy, optimized architecture
- **YOLOv5**: Fast inference, easy deployment
- **Custom Training**: Fine-tuned on specific datasets

## Getting Started

### Prerequisites
```bash
pip install opencv-python torch torchvision
pip install ultralytics  # For YOLOv5
```

### How to Run
Each subdirectory contains specific instructions:

1. **Training Custom Models:**
   - Prepare dataset with annotations
   - Configure YOLO architecture
   - Train on custom data
   - Evaluate performance

2. **Inference:**
   - Load pre-trained weights
   - Run detection on images/video
   - Visualize results

## Key Features
- **Real-time Detection**: Fast inference speeds
- **Custom Training**: Adapt to specific use cases
- **Multiple Classes**: Detect various object types
- **Bounding Boxes**: Precise object localization
- **Confidence Scores**: Detection reliability

## Applications
- Surveillance and security
- Autonomous vehicles
- Retail analytics
- Manufacturing quality control
- Healthcare monitoring
- Smart city infrastructure

## Model Performance
- **Speed**: Real-time detection (30+ FPS)
- **Accuracy**: High mAP scores on custom datasets
- **Robustness**: Works in various lighting/conditions
