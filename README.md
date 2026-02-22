# Resnet-model




#  Image Classification using Transfer Learning (PyTorch + GPU)

## 📌 Project Overview

This project implements an image classification model using **Transfer Learning** in PyTorch.
The model was trained using GPU acceleration and evaluated on both training and test datasets to analyze generalization performance.

The goal was to:

* Implement a full deep learning pipeline
* Use pretrained models effectively
* Apply data augmentation
* Analyze overfitting behavior

---

## 🚀 Technologies Used

* Python
* PyTorch
* Torchvision
* CUDA (GPU training in Google Colab)

---

## 🧩 Key Features

* ✅ Automatic GPU detection using `torch.cuda.is_available()`
* ✅ Transfer Learning with pretrained CNN model
* ✅ Data augmentation using `RandomHorizontalFlip`
* ✅ Batch training with DataLoader
* ✅ Training & Test accuracy evaluation
* ✅ Overfitting analysis (generalization gap)

---

## ⚙️ Model Training Details

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Epochs: (mention how many you used)
* Data Augmentation: Random Horizontal Flip (p=0.5)
* Device: CUDA (GPU)

---

## 📊 Results

| Metric            | Accuracy   |
| ----------------- | ---------- |
| Training Accuracy | **92.9%**  |
| Test Accuracy     | **88.25%** |

### 🔍 Analysis

The model performs slightly better on training data than test data, indicating a small generalization gap (~4–5%).
This suggests mild overfitting, which is expected in deep learning models and within acceptable range.

---

## 🧠 What I Learned

* How GPU acceleration improves training speed
* How transfer learning reduces training time
* Importance of `model.train()` vs `model.eval()`
* Why `torch.no_grad()` is used during evaluation
* How to interpret train vs test accuracy gap

---

## 📂 Project Structure

```
image-classification-project/
│
├── train.py
├── README.md
├── requirements.txt
└── results.png
```

---

## ▶️ How to Run

1. Install dependencies:

```
pip install torch torchvision
```

2. Run the training script:

```
python train.py
```

---

## 📎 Future Improvements

* Add learning rate scheduling
* Implement early stopping
* Experiment with different pretrained models
* Add accuracy/loss graphs per epoch

---




