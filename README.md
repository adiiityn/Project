# Project
# 🍎 Fruit Quality Detection Using Machine Learning

This project uses deep learning to classify fruit images based on their quality (e.g., Good, Bad, Overripe). It combines image processing, model training, and a web interface using Flask.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Flow](#project-flow)
- [1. Data Collection](#1-data-collection)
- [2. Data Preprocessing](#2-data-preprocessing)
- [3. Model Building](#3-model-building)
- [4. Model Training](#4-model-training)
- [5. Model Evaluation](#5-model-evaluation)
- [6. Web Application (Frontend + Backend)](#6-web-application-frontend--backend)
- [7. Running the Project](#7-running-the-project)
- [8. Future Improvements](#8-future-improvements)
- [9. License](#9-license)

---

## ✅ Overview

The Fruit Quality Detection system aims to classify fruit images into categories such as **Good**, **Bad**, or **Overripe** using a Convolutional Neural Network (CNN). The model is served through a Flask API, with a user-friendly HTML frontend.

---

## 🧭 Project Flow

```

Image Dataset → Preprocessing → Model (CNN) → Training → Evaluation → Save Model → Flask App + HTML UI

```

---

## 1. 📷 Data Collection

- You can collect data from:
  - [Kaggle](https://www.kaggle.com)
  - Manually using a phone camera
  - Public datasets (Google Dataset Search)
- Dataset structure:
```

dataset/
├── good/
├── bad/
└── overripe/

````

---

## 2. 🧼 Data Preprocessing

- Resize all images to a consistent shape (e.g., 128x128 pixels).
- Normalize pixel values between 0 and 1.
- Augment data to improve generalization (e.g., flipping, rotation).
- Split dataset into Train, Validation, and Test sets (70/15/15).

Use libraries like `OpenCV`, `Pillow`, `TensorFlow`, or `Keras ImageDataGenerator`.

---

## 3. 🧠 Model Building

We use a **Convolutional Neural Network (CNN)** for image classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
  MaxPooling2D(2,2),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(3, activation='softmax')  # 3 classes: Good, Bad, Overripe
])
````

Compile the model:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 4. 🏋️ Model Training

Train the model using your processed dataset:

```python
model.fit(train_data, epochs=10, validation_data=val_data)
```

Save the trained model:

```python
model.save('fruit_quality_model.h5')
```

---

## 5. 📊 Model Evaluation

* Test the model using unseen data.
* Evaluate using:

  * Accuracy
  * Confusion Matrix
  * Precision / Recall / F1-score
* Visualize training and validation accuracy/loss using `matplotlib`.

---

## 6. 🌐 Web Application (Frontend + Backend)

### 📦 Backend (Flask)

File: `app.py`

* Accepts image uploads
* Preprocesses the image
* Uses the trained model to predict
* Returns the result as JSON

### 🖼 Frontend (HTML + JS)

File: `templates/index.html`

* Simple upload form
* Displays prediction and confidence

---

## 7. 🚀 Running the Project

### 📁 Folder Structure

```
fruit-quality-app/
├── app.py
├── fruit_quality_model.h5
├── templates/
│   └── index.html
```

### 🔧 Requirements

Install dependencies:

```bash
pip install flask tensorflow opencv-python
```

### ▶ Run the server

```bash
python app.py
```

### 🌐 Open in Browser

Go to: `http://127.0.0.1:5000`

Upload an image to get the quality prediction.

## 🚧 Future Improvements

* Add support for more fruit types.
* Use more advanced models (e.g., ResNet, EfficientNet).
* Deploy using Docker or Streamlit Cloud.
* Mobile App Integration using Flutter.
