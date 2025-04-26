# Emotion Recognition using CNN

This project focuses on recognizing human emotions from facial images using Convolutional Neural Networks (CNNs) inspired by the VGG architecture. It supports both binary and multiclass emotion classification and includes a face orientation detection feature.

---

## 🚀 Overview

The goal of this project is to classify emotions based on facial expressions. Two classifiers were developed:

- **Binary Emotion Classifier**: Distinguishes among 4 emotions — `Angry`, `Happy`, `Sad`, and `Surprise`.
- **Multiclass Emotion Classifier**: Extends the binary classifier by adding a `Neutral` class, representing "no emotion".

Additionally, a **Face Orientation Detection** module was trained using a simple deep neural network (DNN) with fully connected layers.

---

## 🧠 Model Architecture

- **Emotion Classifiers**: Based on a VGG-style CNN architecture.
- **Orientation Detector**: A small DNN with Dense layers.

---

## 🗂️ Datasets Used

- **FER-2013**
- **Other Public Facial Expression Datasets**

### Data Preprocessing

- Thorough **manual filtering** of FER-2013 and other datasets to correct misclassified samples.
- **Class rebalancing** and **brightness-based augmentation** to improve performance across varying lighting conditions.

---

## 🎥 Demo

<video width="640" height="360" controls autoplay loop muted>
  <source src="demo_video/emotion_recognition_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 📊 Results

Both classifiers demonstrated good performance after careful dataset curation and augmentation. Face orientation detection ensures better preprocessing before emotion classification.
