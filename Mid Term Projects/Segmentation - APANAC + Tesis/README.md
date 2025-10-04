# Medical Image Segmentation - Cell Nuclei & Tissue Analysis

## Overview
Advanced image segmentation project for medical applications, focusing on cell nuclei segmentation and tissue analysis. Uses U-Net and other segmentation architectures for pixel-level precision.

## Important Notice
**This project is reserved for APANAC and thesis work. It cannot be:**
- Used as a thesis topic
- Presented at JIC or APANAC events

## Project Structure
```
Segmentation - APANAC + Tesis/
├── 1 - Data-Processing.ipynb              # Dataset preparation
├── 2 - Training.ipynb                     # Model training
├── 3 - Predict Mask.ipynb                 # Inference on new images
├── 4 - Evaluation.ipynb                   # Performance metrics
├── 5 - Cell Nuclei Segmentation.ipynb     # Complete pipeline
├── dataset/                               # Training data
├── DSB/                                   # Data Science Bowl dataset
├── prediction/                            # Model outputs
└── test-images/                           # Test samples
```

## Learning Objectives
- Image segmentation fundamentals
- U-Net architecture implementation
- Medical image preprocessing
- Pixel-wise classification
- Evaluation metrics (IoU, Dice coefficient)
- Instance segmentation
- Post-processing techniques

## Getting Started

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib
pip install opencv-python scikit-image pillow
pip install albumentations  # Advanced augmentation
```

### How to Run

**Complete Pipeline (Recommended for beginners):**
```bash
jupyter notebook "5 - Cell Nuclei Segmentation.ipynb"
```

**Step-by-Step Workflow:**
```bash
# Step 1: Prepare data
jupyter notebook "1 - Data-Processing.ipynb"

# Step 2: Train model
jupyter notebook "2 - Training.ipynb"

# Step 3: Generate predictions
jupyter notebook "3 - Predict Mask.ipynb"

# Step 4: Evaluate results
jupyter notebook "4 - Evaluation.ipynb"
```

## Segmentation Tasks

### 1. Cell Nuclei Segmentation
**Applications:**
- Cancer diagnosis
- Cell counting
- Morphology analysis
- Drug response studies

**Challenges:**
- Touching/overlapping cells
- Variable cell sizes
- Staining variations
- Background noise

### 2. Tissue Segmentation
**Applications:**
- Organ delineation
- Tumor boundary detection
- Tissue classification
- Surgical planning

## Dataset

### Data Science Bowl 2018
- 670+ nuclei images
- Various tissues and staining
- Instance-level annotations
- Ideal for learning segmentation

### Data Structure
```
dataset/
├── images/
│   ├── train/
│   └── test/
└── masks/
    ├── train/
    └── test/
```

## U-Net Architecture

### Why U-Net?
- Designed for medical image segmentation
- Works with small datasets
- Symmetric encoder-decoder
- Skip connections preserve details
- State-of-the-art for biomedical images

### Architecture Components
```
Input Image (256x256x3)
↓
Encoder (Downsampling):
- Conv → ReLU → Conv → ReLU → MaxPool (×4)
- Captures context

Bottleneck:
- Conv → ReLU → Conv → ReLU

Decoder (Upsampling):
- UpConv → Concatenate with encoder → Conv → ReLU (×4)
- Precise localization

Output:
- Conv 1×1 → Sigmoid
- Binary mask (256x256x1)
```

## Workflow Details

### 1. Data Processing
- Load images and masks
- Resize to uniform dimensions
- Normalize pixel values
- Create train/validation split
- Data augmentation:
  - Rotation, flipping
  - Elastic deformation
  - Brightness/contrast
  - Gaussian noise

### 2. Training
- **Loss Function**: Binary Cross-Entropy + Dice Loss
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 8-16
- **Epochs**: 50-100
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

### 3. Prediction
- Load trained model
- Process test images
- Generate segmentation masks
- Post-processing:
  - Threshold binary masks
  - Morphological operations
  - Instance separation

### 4. Evaluation
**Metrics:**
- **IoU (Jaccard Index)**: Overlap between prediction and ground truth
- **Dice Coefficient**: 2×overlap / (pred + truth)
- **Pixel Accuracy**: Correctly classified pixels
- **Precision/Recall**: For binary segmentation

## Advanced Techniques

### Instance Segmentation
Separate individual touching cells:
- Watershed algorithm
- Distance transform
- Contour detection
- Marker-based separation

### Post-Processing
- Remove small artifacts
- Fill holes
- Smooth boundaries
- Connected component analysis

### Data Augmentation
```python
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate,
    ElasticTransform, GridDistortion,
    RandomBrightnessContrast
)
```

## Model Variants

### U-Net Improvements
- **U-Net++**: Nested U-Net with dense connections
- **Attention U-Net**: Attention gates for focus
- **ResUNet**: Residual blocks
- **U-Net with EfficientNet encoder**: Better features

### Other Architectures
- **Mask R-CNN**: Instance segmentation
- **DeepLab**: Atrous convolution
- **FCN**: Fully Convolutional Networks

## Evaluation Metrics Explained

### IoU (Intersection over Union)
```
IoU = Area of Overlap / Area of Union
Range: 0 to 1 (higher is better)
```

### Dice Coefficient
```
Dice = 2 × |Pred ∩ Truth| / (|Pred| + |Truth|)
Range: 0 to 1 (higher is better)
More forgiving to small errors than IoU
```

## Applications

### Medical Diagnostics
- Cancer cell identification
- Tumor boundary delineation
- Organ segmentation for surgery
- Disease progression tracking

### Research
- Cell proliferation studies
- Drug efficacy testing
- Morphological analysis
- High-throughput screening

### Pathology
- Digital pathology
- Automated cell counting
- Tissue classification
- Quality control

## Best Practices
- Use data augmentation extensively
- Monitor both IoU and Dice
- Visualize predictions frequently
- Test on diverse samples
- Ensemble multiple models
- Fine-tune on domain-specific data
- Post-process for cleaner results

## Common Challenges & Solutions

### Touching Cells
**Solution:**
- Add distance transform branch
- Watershed post-processing
- Instance segmentation models

### Class Imbalance
**Solution:**
- Weighted loss functions
- Focal loss
- Balanced batch sampling

### Small Objects
**Solution:**
- Higher resolution inputs
- Multi-scale predictions
- Attention mechanisms

### Limited Data
**Solution:**
- Strong augmentation
- Transfer learning
- Synthetic data generation

## Future Enhancements
- 3D segmentation for volumetric data
- Real-time segmentation
- Active learning for annotation
- Weakly supervised methods
- Self-supervised pre-training
- Federated learning for privacy

## Clinical Integration
- DICOM compatibility
- Integration with microscopy systems
- Automated reporting
- Quality metrics dashboard
- Batch processing capabilities
