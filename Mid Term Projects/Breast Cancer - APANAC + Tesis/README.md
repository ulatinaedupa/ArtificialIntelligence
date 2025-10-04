# Breast Cancer Detection - Deep Learning Project

## Overview
Advanced deep learning project for breast cancer classification using medical imaging. This project includes complete data processing, model training, and a deployment-ready Gradio web application for real-time predictions.

## Important Notice
**This project is reserved for APANAC and thesis work. It cannot be:**
- Used as a thesis topic
- Presented at JIC or APANAC events

## Project Structure
```
Breast Cancer - APANAC + Tesis/
├── Breast Cancer - Deep Learning.ipynb    # Main training notebook
├── Breast Cancer - Gradio App.ipynb       # Gradio app development
├── gradio_deploy.py                       # Standalone deployment script
└── example/                               # Sample data and models
```

## Getting Started

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib
pip install scikit-learn opencv-python pillow gradio
```

### How to Run

**1. Training the Model:**
```bash
jupyter notebook "Breast Cancer - Deep Learning.ipynb"
```

**2. Gradio App Development:**
```bash
jupyter notebook "Breast Cancer - Gradio App.ipynb"
```

**3. Deploy Standalone Application:**
```bash
python gradio_deploy.py
```
Access the app at http://localhost:7860

## Key Features
- Transfer learning with pre-trained models (VGG16, ResNet50)
- Data augmentation for robust training
- High sensitivity for cancer detection
- Gradio web interface for easy deployment
- Real-time predictions with confidence scores
- Visualization of model attention (Grad-CAM)

## Model Performance Goals
- **Sensitivity**: >95% (minimize missed cancer cases)
- **Specificity**: >90% (reduce false alarms)
- **AUC-ROC**: >0.95

## Clinical Considerations
This is a screening tool to assist medical professionals, not replace them. Always require human verification for clinical decisions.
