# 🚗 Lane Detection: Two Approaches (OpenCV & ENet)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

This project explores **two different approaches** for detecting lane markings from road images/videos:

1. **Classical Computer Vision (OpenCV + Hough Transform)**  
2. **Deep Learning (ENet Semantic Segmentation)**  

The goal is to compare traditional image processing pipelines with modern deep learning models in the context of **driver assistance and autonomous driving systems**.

---

## 📖 Overview

### 🔹 Approach 1: Classical Computer Vision

This approach uses **traditional image processing techniques** in OpenCV to detect lane lines without the need for training data.  
It follows a step-by-step pipeline:

#### 📌 Step 1: Grayscale Conversion
- Road frames are first converted from RGB to **grayscale**.  
- This reduces computational complexity since color information is not necessary for edge-based lane detection.  

#### 📌 Step 2: Edge Detection (Canny)
- The **Canny Edge Detector** is applied to highlight strong intensity gradients in the frame.  
- Lane markings usually have sharp contrast with the road surface, making them stand out as edges.  

#### 📌 Step 3: Region of Interest (ROI)
- Since lanes usually appear in the lower part of the frame, a **polygonal mask** is defined to focus only on the road area.  
- This removes irrelevant details (sky, trees, vehicles, etc.) and reduces false detections.  

#### 📌 Step 4: Hough Line Transform
- The **Probabilistic Hough Transform** is applied on the detected edges to identify straight line segments.  
- These line segments correspond to **left and right lane boundaries**.  
- Post-processing like slope filtering is used to distinguish between left and right lanes.  

---

#### 🎥 Demonstration
A demo video showing this pipeline applied to real driving footage is included in the repository:  

[▶️ Watch Classical CV Lane Detection Demo](video_demo_classical.mp4)

---

#### ✅ Advantages
- **Lightweight & Fast:** Runs in real-time on CPU.  
- **Explainable:** Each step of the pipeline is easy to visualize and debug.  

#### ⚠️ Limitations
- **Sensitive to Noise:** Shadows, road texture, and cracks can cause false edges.  
- **Poor Generalization:** Struggles in rainy/night conditions.  
- **Assumes Straight Lanes:** Curved or occluded lanes are often misdetected.  

---

### 🔹 Approach 2: Deep Learning with ENet

This approach leverages **deep learning** for lane detection by using the **ENet (Efficient Neural Network)** architecture for **semantic segmentation**.  
Instead of detecting edges and lines explicitly, the model learns to classify each pixel in the image as either **lane** or **background**.

---
- **Notebook Link:** [Binayak Dey](https://www.kaggle.com/code/binayakdey/lane-detection-model-enet)  

#### 📌 Step 1: Dataset Preparation
- A custom **Road Lane Estimation dataset** was created and labeled manually using [Roboflow](https://roboflow.com).  
- The dataset includes:
  - **Input Images:** Road scenes captured from a driver’s perspective  
  - **Segmentation Masks:** Pixel-level annotations where lane markings are highlighted  

---

#### 📌 Step 2: Model Architecture (ENet)
- **ENet** is a lightweight semantic segmentation model designed for **real-time applications**.  
- Key features:
  - **Encoder-Decoder design:** Encodes spatial features, then upsamples to produce segmentation masks  
  - **Early downsampling:** Reduces computation by aggressively shrinking feature maps early  
  - **Bottleneck modules:** Capture both local and global context efficiently  
- Well-suited for real-time tasks like autonomous driving where speed matters.  

---

#### 📌 Step 3: Training
- Framework: **PyTorch**  
- Loss Function: **Cross-Entropy Loss**  
- Optimizer: **Adam**  
- Training/Validation split applied to ensure generalization  
- Trained for multiple epochs on GPU until convergence  

Full training code is available in the Kaggle notebook:  
👉 [Lane Detection Model (ENet) - Kaggle](https://www.kaggle.com/code/binayakdey/lane-detection-model-enet/edit)

---

#### 📌 Step 4: Inference & Visualization
- The trained ENet model outputs **binary masks** where lane pixels are highlighted.  
- Post-processing overlays these masks on the original frame to visualize detected lanes.  

Example visualization:  
![ENet Lane Detection](result_enet.png)  
*(Left: Input Image | Right: ENet Predicted Mask)*  

---

#### ✅ Advantages
- **Robust:** Handles noise, shadows, and varying lighting better than classical methods  
- **Pixel-level precision:** Can detect complex curved lanes and multiple markings  
- **Scalable:** Performance improves with more labeled data  

#### ⚠️ Limitations
- **Data hungry:** Requires a labeled dataset for training  
- **Compute intensive:** Training needs a GPU for efficiency  
- **Generalization challenges:** May fail in unseen conditions unless dataset is diverse  

---

## 📂 Dataset
- **Name:** Road Lane Estimation  
- **Source:** [Roboflow Dataset](https://app.roboflow.com/binayak-dey-oo1z9/rode-lane-estimation-c8te6/3)  

---

## ⚙️ Training Details (ENet)
- Framework: **PyTorch**  
- Model: **ENet**  
- Loss: Cross-Entropy Loss  
- Optimizer: Adam  
- Dataset split: Train / Validation  

Training notebook:  
👉 [Lane Detection Model (ENet) - Kaggle](https://www.kaggle.com/code/binayakdey/lane-detection-model-enet/edit)

---

## 🖼️ Results

Both approaches were tested and compared:

### Classical CV (Hough Transform)
![Classical Lane Detection](result_classical.png)  
*(Example using OpenCV edge + Hough Transform pipeline)*  

### Deep Learning (ENet Segmentation)
![ENet Lane Detection](result_enet.png)  
*(Left: Input | Right: ENet Predicted Lane Mask)*  

---

## 📌 Future Work
- Train and compare **SCNN, SegNet, and ENet** for lane segmentation  
- Extend dataset to diverse conditions (night, rain, occlusion)  
- Real-time integration with **YOLOv8 object detection** for a complete perception system  

---

## 🙏 Acknowledgements
- Dataset labeled via [Roboflow](https://roboflow.com)  
- ENet architecture reference: *"ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"*  

---
