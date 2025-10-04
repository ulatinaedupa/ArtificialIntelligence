# Project 17: MLP, CNN & LSTM in TensorFlow Lite

## Overview
Learn to build, train, and deploy neural networks (MLP, CNN, LSTM) optimized for mobile and embedded devices using TensorFlow Lite.

## Learning Objectives
- Multi-Layer Perceptron (MLP) architecture
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM) networks
- TensorFlow Lite conversion
- Mobile ML deployment
- Model optimization for edge devices

## Neural Network Architectures

### 1. MLP (Multi-Layer Perceptron)
- **Use Cases**:
  - Tabular data classification
  - Simple regression tasks
  - Feature-based predictions
- **Structure**: Fully connected layers

### 2. CNN (Convolutional Neural Network)
- **Use Cases**:
  - Image classification
  - Object detection
  - Computer vision tasks
- **Structure**: Conv layers + pooling + dense

### 3. LSTM (Long Short-Term Memory)
- **Use Cases**:
  - Time series prediction
  - Sequence modeling
  - Natural language processing
- **Structure**: Recurrent layers with memory cells

## Getting Started

### Prerequisites
```bash
pip install tensorflow numpy matplotlib
pip install scikit-learn pandas
```

### How to Run
```bash
jupyter notebook  # Open relevant notebooks
# Or run Python scripts for training
```

## TensorFlow Lite Workflow

### 1. Build & Train Model
```python
# Create model in TensorFlow/Keras
model = Sequential([...])
model.compile(...)
model.fit(X_train, y_train)
```

### 2. Convert to TFLite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 3. Optimize
- Quantization (INT8, FP16)
- Pruning
- Weight clustering
- Model compression

### 4. Deploy
- Android/iOS apps
- Raspberry Pi
- Microcontrollers
- Edge devices

## Key Features
- **Model Compression**: Reduce size for mobile
- **Quantization**: INT8 for faster inference
- **Cross-Platform**: Android, iOS, embedded
- **Low Latency**: Optimized for edge
- **Offline Capability**: No cloud dependency

## Applications

### MLP Applications
- Customer churn prediction
- Credit scoring
- Disease diagnosis
- Recommendation systems

### CNN Applications
- Image classification
- Face recognition
- Medical imaging
- Quality inspection

### LSTM Applications
- Stock price prediction
- Language modeling
- Anomaly detection
- Gesture recognition

## Optimization Techniques
- **Quantization**: Reduce precision
- **Pruning**: Remove unnecessary weights
- **Clustering**: Group similar weights
- **Knowledge Distillation**: Teacher-student models

## Deployment Platforms
- Android (Java/Kotlin)
- iOS (Swift)
- Raspberry Pi
- Arduino
- Web (TensorFlow.js)

## Performance Metrics
- Model size (MB)
- Inference time (ms)
- Accuracy trade-offs
- Power consumption
- Memory usage

## Use Cases
- Mobile apps with ML
- IoT devices
- Real-time processing
- Privacy-focused ML (on-device)
- Offline applications
- Resource-constrained environments
