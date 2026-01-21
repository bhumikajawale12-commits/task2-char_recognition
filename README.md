# Handwritten Character Recognition using CNN

## 📌 Project Overview

This project implements a **Handwritten Character Recognition System** using a **Convolutional Neural Network (CNN)**.
The model is trained on the **MNIST dataset** to recognize handwritten digits (0–9) and can also predict custom handwritten digit images.

---

## 🎯 Objective

To build a deep learning model that accurately identifies handwritten digits using image classification techniques.

---

## 🧠 Technologies Used

- Python  
- TensorFlow & Keras  
- NumPy  
- OpenCV  
- Scikit-learn  
- Matplotlib  


---

## 📊 Dataset

**MNIST Handwritten Digit Dataset**

- 60,000 training images  
- 10,000 testing images  
- Image size: 28×28 pixels  
- Grayscale images of digits (0–9)

---

## ⚙️ Model Architecture

- Conv2D (32 filters, 3×3, ReLU)
- MaxPooling (2×2)
- Conv2D (64 filters, 3×3, ReLU)
- MaxPooling (2×2)
- Flatten Layer
- Dense (128 neurons, ReLU)
- Output Dense (10 neurons, Softmax)

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies

### 2️⃣ Train the model

### 3️⃣ Evaluate the model

### 4️⃣ Predict custom handwritten digit
Place an image named `2.png` in the project folder, then run:


---

## ✅ Results

- Training Accuracy: ~98%  
- Successfully predicts handwritten digits  
- Supports custom image testing  



---

## 🔮 Future Scope

- Extend to handwritten alphabet recognition (EMNIST)
- Build a web app for real-time digit drawing
- Improve accuracy with deeper CNN models

---

## 👩‍💻 Author

**Bhumika Jawale**

---

## 📌 Acknowledgement

MNIST Dataset by Yann LeCun  
TensorFlow & Keras Libraries


