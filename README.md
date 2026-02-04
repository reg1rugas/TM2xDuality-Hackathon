# TM2xDuality-Hackathon

> **Pixel-wise semantic segmentation for autonomous navigation in unstructured off-road environments**

Deep learning solution for classifying desert terrain into 10 distinct categories using UNet with ResNet34 encoder. Achieves **0.60 validation mIoU** and **0.42 test mIoU** through strategic handling of severe class imbalance, and uses an A* algorithm for minimal cost pathfinding.


# Phase 1 : Off-Road Terrain Semantic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Results](#results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Problem Statement

Autonomous vehicles require pixel-level understanding of off-road terrains for safe navigation in unstructured environments. This project addresses semantic segmentation of desert landscapes with 10 terrain categories:

| Class ID | Category | Description | Avg. Presence |
|----------|----------|-------------|---------------|
| 0 | Trees | Tall vegetation | 0.38% |
| 1 | Lush Bushes | Green shrubs | <0.01% |
| 2 | Dry Grass | Brown/yellow grass | 23.45% |
| 3 | Dry Bushes | Dead shrubs | 2.06% |
| 4 | Ground Clutter | Small debris | 12.88% |
| 5 | Flowers | Flowering plants | 1.13% |
| 6 | Logs | Fallen wood | 0.01% |
| 7 | Rocks | Stones/boulders | 18.97% |
| 8 | Landscape | Ground/terrain | 40.48% |
| 9 | Sky | Sky regions | 14.65% |

**Key Challenge**: Severe class imbalance with some classes representing <0.01% of pixels.

---

## ✨ Features

- 🎯 **38% improvement** over baseline (0.23 → 0.60 mIoU)
- ⚡ **Real-time**: 18.22ms inference on Tesla T4 GPU (54.89 FPS)
- 🔄 **Class imbalance handling**: Weighted loss + 3× oversampling
- 🚀 **Multi-GPU training**: DataParallel support with mixed precision (FP16)
- 📊 **Comprehensive metrics**: Per-class IoU, precision, recall, F1-score
- 💾 **Auto-checkpointing**: Best model + periodic saves
- 📈 **Real-time monitoring**: Progress bars, loss curves, class statistics
- 🔍 **Production-ready**: Complete inference pipeline with visualization

---

## 📊 Results

### Overall Performance

| Metric | Validation | Test |
|--------|------------|------|
| **Mean IoU** | **0.60** | **0.42** |
| **Pixel Accuracy** | 88.05% | 61.68% |
| **Training Time** | 4 hours | - |
| **Inference Speed** | 122ms/image | 122ms/image |

### Per-Class Performance (Validation)

| Class | IoU | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| **Sky** | **0.9833** | 0.9858 | 0.9974 | 0.9915 |
| **Trees** | **0.7048** | 0.8102 | 0.8365 | 0.8231 |
| **Dry Grass** | **0.6701** | 0.7821 | 0.8123 | 0.7969 |
| **Landscape** | **0.6564** | 0.8018 | 0.7797 | 0.7897 |
| Rocks | 0.5055 | 0.6234 | 0.7456 | 0.6789 |
| Dry Bushes | 0.2946 | 0.5892 | 0.3904 | 0.4337 |
| Flowers | 0.2324 | 0.3012 | 0.6234 | 0.4056 |
| Logs | 0.3030 | 0.3845 | 0.6789 | 0.4912 |

**Note**: Some classes (Lush Bushes, Ground Clutter, Logs) are extremely rare or absent in test set.

---

## 🏗️ Architecture

### Model: UNet + ResNet34

```
Input Image (960×544×3)
         ↓
┌─────────────────────────┐
│  ResNet34 Encoder       │
│  (ImageNet Pretrained)  │
│  • Conv layers 1-4      │
│  • Feature extraction   │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  UNet Decoder           │
│  • Upsampling blocks    │
│  • Skip connections     │
│  • Progressive fusion   │
└─────────────────────────┘
         ↓
Output Mask (960×544×10)
```

### Loss Function

**Combined Loss** = 0.75 × Dice Loss + 0.25 × Weighted Cross-Entropy

- **Dice Loss**: Optimizes IoU directly, handles class imbalance
- **Weighted CE**: Class weights inversely proportional to frequency
- **Class Weights**: `[1.0, 3.0, 1.0, 2.0, 1.5, 3.0, 3.0, 1.5, 1.0, 1.0]`

### Training Strategy

1. **Data Balancing**: 3× oversampling of rare classes (Dry Bushes, Flowers)
2. **Augmentation**: Rotation (±15°), flip, brightness (±20%), contrast (±20%)
3. **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
4. **Class weights**: Adding apt. weights for rare items to force learning

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM
- 8GB+ GPU VRAM (Tesla T4/V100 recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/reg1rugas/TM2xDuality-Hackathon
cd TM2xDuality-Hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install segmentation_models_pytorch
pip install segmentation-models-pytorch
```

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

---

## 📁 Dataset Preparation

### Expected Structure

```
data/
├── train/
│   ├── images/          # RGB images (960×540 JPG)
│   └── masks/           # Segmentation masks (960×540 PNG)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Class Mapping

Mask pixel values → Class indices:

```python
CLASS_MAPPING = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    600: 4,    # Ground Clutter
    1000: 5,   # Flowers
    5000: 6,   # Logs
    7100: 7,   # Rocks
    8000: 8,   # Landscape
    10000: 9   # Sky
}
```

### Data Statistics

- **Training set**: 2857 images
- **Validation set**: 317 images
- **Test set**: 1002 images
- **Image resolution**: 960×540 (resized to 960×544 for training)

---

## 🚀 Training

### Quick Start

```bash
# Basic training (single GPU)
python train.py --data_dir ./data --output_dir ./outputs

# Multi-GPU training
python train.py --data_dir ./data --gpus 0,1,2,3

# Resume from checkpoint
python train.py --resume ./outputs/checkpoints/best_model.pth
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Root directory containing train/val/test |
| `--output_dir` | `./outputs` | Output directory for checkpoints & logs |
| `--encoder` | `resnet34` | Encoder architecture (resnet34/resnet50/efficientnet-b0) |
| `--batch_size` | `8` | Batch size per GPU |
| `--epochs` | `100` | Maximum training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--img_height` | `544` | Input image height |
| `--img_width` | `960` | Input image width |
| `--num_workers` | `4` | DataLoader workers |
| `--mixed_precision` | `False` | Enable FP16 training |
| `--gpus` | `0` | GPU IDs (comma-separated) |
| `--resume` | `None` | Resume from checkpoint path |
| `--early_stopping_patience` | `20` | Early stopping patience |
| `--save_every` | `5` | Save checkpoint every N epochs |

### Training Output (in mounted Drive)

```
checkpoints/
├── best_model.pth           # Highest mIoU Model 
├── checkpoint_epoch_10.pth
└── logs/
    ├── training_log.csv     # Metrics (Loss, IoU, Accuracy)
    └── config.json          # Hyperparameters used
```

### Monitoring Training

```bash
# View live metrics
tail -f checkpoints/logs/training_log.csv

# TensorBoard (Optional)
tensorboard --logdir checkpoints/logs
```

---

## 🔮 Inference

### Quick Inference

```bash
# Single image
python test.py \
    --checkpoint ./outputs/checkpoints/best_model.pth \
    --image ./test_image.jpg \
    --output ./prediction.png

# Batch inference on test set
python test.py \
    --checkpoint ./outputs/checkpoints/best_model.pth \
    --test_dir ./data/test/images \
    --output_dir ./predictions \
    --visualize
```

### Inference Script

```python
import torch
import cv2
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# 1. Load Duality Model
model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=10)
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# 2. Preprocessing
transform = Compose([
    Resize(544, 960),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 3. Inference
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed = transform(image=image)
input_tensor = transformed['image'].unsqueeze(0).cuda()

with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).cpu().numpy()[0]

# 4. Save
cv2.imwrite('prediction.png', prediction.astype('uint8'))
```

### Latency Benchmarking

```bash
# Run latency benchmark
python test.py \
    --checkpoint ./outputs/checkpoints/best_model.pth \
    --benchmark \
    --num_runs 100
```

**Target Performance (Tesla T4):**
```
~18.2 ms (54 FPS)
```

---

## ⚙️ Configuration

### TrainingConfig Class

```python
class TrainingConfig:
    # Architecture
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES = 10 
    
    # Input Dimensions
    IMG_HEIGHT = 544
    IMG_WIDTH = 960

    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Loss Weights (Handling Imbalance: Sky vs Logs)
    # [Trees, Bushes, Grass, DryBush, Clutter, Flowers, Logs, Rocks, Land, Sky]
    CLASS_WEIGHTS = [1.0, 3.0, 1.0, 2.0, 1.5, 5.0, 8.0, 1.5, 1.0, 1.0]

    # Augmentation
    ROTATION_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.2
    
    # System
    MIXED_PRECISION = True
    NUM_WORKERS = 4
```

## 🔬 Technical Details

### Class Imbalance Handling

**Problem**: Sky (15%) vs Logs (0.01%) = 1500× imbalance

**Solution**:
1. **Weighted Loss**: Inverse frequency weighting
2. **Oversampling**: Sample rare classes 3× more frequently
3. **Dice Loss**: Penalizes false negatives on small classes

### Data Augmentation Pipeline

```python
train_transform = A.Compose([
    A.Resize(544, 960),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 4

# Enable mixed precision
python train.py --mixed_precision

# Reduce image resolution
python train.py --img_height 480 --img_width 848
```

#### 2. Poor Performance on Rare Classes

```python
# Increase class weights
CLASS_WEIGHTS = [1.0, 5.0, 1.0, 3.0, 2.0, 5.0, 5.0, 2.0, 1.0, 1.0]

# Increase oversampling
OVERSAMPLE_FACTOR = 5
```

#### 3. Slow Training

```bash
# Increase num_workers
python train.py --num_workers 8

# Enable mixed precision
python train.py --mixed_precision

# Use multiple GPUs
python train.py --gpus 0,1,2,3
```

#### 4. Model Not Loading (PyTorch 2.6+)

```python
# Add weights_only=False
checkpoint = torch.load(path, weights_only=False)
```

---
<br><br>

# Phase 2 : Elective Track (Computer Vision (Path Planning))

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


The **Neural Nav System** is a visualization tool designed to bridge the gap between deep learning perception and navigational logic. It wraps a **ResNet34-UNet** segmentation engine in a sci-fi "Glassmorphism" HUD, allowing researchers to simulate autonomous agents and analyze A* pathfinding decisions in real-time.

---

## 📋 Table of Contents

- [System Overview](#-system-overview)
- [Key Features](#-key-features)
- [Installation (Critical)](#-installation-critical)
- [Usage Guide](#-usage-guide)
---

## 🔭 System Overview

1.  **Perception**: Runs inference on raw terrain images to generate semantic cost maps.
2.  **Planning**: Executes A* pathfinding algorithms to find the optimal route between user-defined points.
3.  **Telemetry**: Logs inference latency, path efficiency, and total traversal cost.

The UI is styled with a **"Deep Space"** aesthetic, featuring **Orbitron** typography, neon accents, and a reactive canvas system.

---

## ✨ Key Features

* **"Bake & Clear" Engine**: Implements a zero-latency drawing system where completed paths are "baked" into the background map, allowing for multiple agents (Unit 0-4) to be deployed sequentially without performance loss.
* **Smart Resizing**: Automatically handles input images of varying aspect ratios, snapping them to grid multiples (32px) required by the UNet architecture.
* **Safety Snapping**: If a user clicks on an obstacle (e.g., a rock), the system automatically snaps the waypoint to the nearest safe pixel within a 20px radius.
* **Debug Layers**: Toggleable views for **Raw Cost Maps** and **Thermal Heatmaps** to inspect model confidence.

---

## 🛠 Installation (Critical)

⚠️ **System Warning**: This application relies on specific canvas event handling that was altered in newer Streamlit versions. **You must use Streamlit 1.33.0.**

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install the exact Streamlit version required
pip install streamlit==1.33.0

# Install Computation & AI libs
pip install torch torchvision segmentation-models-pytorch numpy opencv-python-headless pandas

# Install UI libs
pip install streamlit-drawable-canvas packaging
```

## 🚀 Usage Guide

### 1. Launch the streamlit app
```bash
streamlit run app.py
```

### 2. Initialization Sequence
1. **Upload Weights**: Drag and drop your trained .pth file (ResNet34-UNet) into the sidebar.

2. **Upload Terrain**: Upload a raw RGB image (.jpg, .png).

3. **System Check**: The app will run segmentation inference (~18ms on GPU) and generate the navigational graph.

### 3. Deployment (Interactive)
The system supports up to 5 unique agents with distinct color codes:
- Step 1: Click on the map to set the Start Point.
- Step 2: Click a second location to set the Goal.
- Step 3: The system calculates the path, draws it on the map, and logs the telemetry.Repeat: The agent counter increments (Unit 0 $\to$ Unit 1), and you can plot the next path immediately.
- ResetClick the REBOOT SYSTEM button in the sidebar to clear all paths and reset metrics.

## 🧭 Navigational Logic
The A* planner uses a Manhattan Distance heuristic ($h = |dx| + |dy|$) and interprets the terrain based on the following cost table:

| Class ID | Terrain Type | Cost | Behavior |
| :--- | :--- | :--- | :--- |
| **0, 1, 6, 7** | Trees, Bushes, Logs, Rocks | **255** | **Hard Obstacle** (Impassable) |
| **3, 4** | Dry Bushes, Clutter | **10-20** | **High Cost** (Avoid if possible) |
| **5** | Flowers | **5** | **Medium Cost** (Soft avoid) |
| **2, 8** | Dry Grass, Landscape | **1** | **Free Space** (Preferred) |

## 🐛 Troubleshooting

### "Black Screen" on the Map Canvas

- Cause: You are likely running Streamlit v1.35.0+.

- Fix: Downgrade immediately: pip install streamlit==1.33.0.

### "Coordinate Error: Unsafe Terrain"

- Cause: You clicked deep inside a large obstacle (Rock/Tree) where the safety snapper couldn't find a safe point within 20 pixels.

- Fix: Try clicking closer to the edge of the obstacle or on open ground.
