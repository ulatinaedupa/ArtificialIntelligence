# Lab 3: Custom Object Detection with YOLOv5

## Overview
Advanced computer vision laboratory focused on training custom object detection models using YOLOv5. Learn to prepare datasets, train models, and deploy object detectors for specific use cases.

## Project Structure
```
Lab3/
├── Custom_object_detection_yolov5.ipynb    # Main lab notebook
├── datasets/                                # Custom datasets
├── models/                                  # Trained weights
├── configs/                                 # YOLO configurations
└── results/                                 # Detection outputs
```

## Learning Objectives
- Object detection fundamentals
- YOLOv5 architecture understanding
- Custom dataset preparation
- Data annotation techniques
- Model training from scratch/transfer learning
- Performance evaluation (mAP, IoU)
- Real-time inference
- Model optimization and deployment

## Getting Started

### Prerequisites
```bash
pip install torch torchvision opencv-python
pip install ultralytics  # YOLOv5
pip install labelImg  # For annotation
pip install matplotlib pillow
```

### How to Run
```bash
jupyter notebook Custom_object_detection_yolov5.ipynb
```

## Lab Workflow

### 1. Dataset Preparation
**Tasks:**
- Collect images for your use case
- Annotate images with bounding boxes
- Organize dataset structure
- Split data (train/val/test)

**Required Format:**
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**Annotation Tools:**
- LabelImg (recommended)
- Roboflow
- CVAT
- VGG Image Annotator

### 2. Data Annotation
- Draw bounding boxes around objects
- Assign class labels
- Export in YOLO format
- Verify annotations

**YOLO Label Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```

### 3. Configuration
Create `data.yaml`:
```yaml
train: path/to/train/images
val: path/to/val/images
nc: 3  # number of classes
names: ['class1', 'class2', 'class3']
```

### 4. Model Training
**From Scratch:**
```python
!python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights ''
```

**Transfer Learning (Recommended):**
```python
!python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```

**Training Parameters:**
- `--img`: Input image size
- `--batch`: Batch size
- `--epochs`: Training iterations
- `--weights`: Pre-trained weights or '' for scratch
- `--device`: GPU/CPU selection

### 5. Model Evaluation
**Metrics:**
- **mAP@.5**: Mean Average Precision at IoU 0.5
- **mAP@.5:.95**: mAP across IoU thresholds
- **Precision**: Correct detections / All detections
- **Recall**: Detected objects / All objects
- **Inference Speed**: FPS on target hardware

**Evaluation Tools:**
- Confusion matrix
- PR curves
- F1-confidence curves
- Detection visualizations

### 6. Inference & Testing
```python
# Detect on images
!python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images

# Detect on video
!python detect.py --weights best.pt --source path/to/video.mp4

# Real-time webcam
!python detect.py --weights best.pt --source 0
```

## YOLOv5 Architecture

### Model Variants
- **YOLOv5n**: Nano (fastest, smallest)
- **YOLOv5s**: Small (balanced)
- **YOLOv5m**: Medium
- **YOLOv5l**: Large
- **YOLOv5x**: Extra-large (most accurate)

### Key Components
- **Backbone**: CSPDarknet53
- **Neck**: PANet
- **Head**: YOLO detection layer
- **Loss**: Multi-part (box, objectness, classification)

## Custom Use Cases Examples

### 1. Safety Monitoring
- Detect PPE (helmets, vests, goggles)
- Identify safety violations
- Real-time alerts

### 2. Inventory Management
- Product detection on shelves
- Count items
- Detect missing stock

### 3. Quality Control
- Defect detection
- Part identification
- Assembly verification

### 4. Traffic Monitoring
- Vehicle detection and counting
- License plate recognition
- Parking management

### 5. Agriculture
- Crop disease detection
- Fruit counting
- Weed identification

## Performance Optimization

### Data Augmentation
- Mosaic augmentation
- Random scaling
- Color jitter
- Horizontal flip
- Crop and pad

### Training Tricks
- Progressive resizing
- Learning rate scheduling
- Early stopping
- Ensemble models
- Test-time augmentation

### Inference Optimization
- **Model Export**: ONNX, TensorRT, CoreML
- **Quantization**: INT8 for speed
- **Pruning**: Reduce model size
- **Batch Inference**: Process multiple images

## Deployment Options
- **Edge Devices**: Raspberry Pi, Jetson Nano
- **Mobile**: iOS/Android apps
- **Cloud**: AWS, GCP, Azure
- **Web**: TensorFlow.js, ONNX.js
- **Desktop**: Python application

## Lab Deliverables
1. Annotated custom dataset
2. Trained YOLOv5 model
3. Evaluation metrics and plots
4. Detection visualizations
5. Inference code
6. Model weights (.pt file)
7. Documentation of process

## Common Challenges & Solutions

### Challenge: Low mAP
**Solutions:**
- Collect more diverse data
- Improve annotation quality
- Increase training epochs
- Try data augmentation
- Use larger model variant

### Challenge: Overfitting
**Solutions:**
- Add more training data
- Increase augmentation
- Use regularization
- Reduce model complexity
- Early stopping

### Challenge: Slow Inference
**Solutions:**
- Use smaller model (YOLOv5n/s)
- Reduce input size
- Export to ONNX/TensorRT
- Batch processing
- Use GPU acceleration

### Challenge: False Positives
**Solutions:**
- Increase confidence threshold
- Improve training data quality
- Balance classes
- Add hard negative mining

## Best Practices
- Start with pre-trained weights
- Use high-quality annotations
- Balance your dataset
- Monitor validation metrics
- Save checkpoints regularly
- Test on diverse images
- Document your process
- Version your datasets and models

## Tips for Success
- Minimum 100-300 images per class
- Diverse backgrounds and lighting
- Consistent annotation quality
- Validate annotations before training
- Use transfer learning
- Monitor training curves
- Test on real-world scenarios
- Iterate based on results

## Resources
- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [Roboflow Datasets](https://universe.roboflow.com/)
- [Computer Vision Tutorials](https://www.ultralytics.com/tutorials)
