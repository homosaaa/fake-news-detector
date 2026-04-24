# 📰 Fake News Detection System (ML vs RNN vs LSTM)

---

## 📌 Project Overview

This project is an NLP-based Fake News Detection System that compares three approaches:

- Machine Learning Model (TF-IDF + Logistic Regression)
- Deep Learning Model (RNN)
- Deep Learning Model (LSTM)

The goal is to classify news articles into:

- 0 → Fake News  
- 1 → Real News  

And compare traditional Machine Learning vs Deep Learning models.

---

## 📂 Dataset

Dataset used from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains:

- Fake.csv → Fake news articles  
- True.csv → Real news articles  

---

## ⚙️ Project Workflow

- Load dataset
- Merge title + text
- Label encoding
- Shuffle data
- Text preprocessing
- Train/Test split
- Model training
- Evaluation

---

# 🧠 Models

---

## 📌 1. Machine Learning Model (Logistic Regression)

- TF-IDF Vectorization
- Logistic Regression classifier

### Steps:
- Text cleaning
- TF-IDF feature extraction
- Train/Test split
- Model training
- Prediction

---

## 📌 2. Deep Learning Model (RNN)

- Simple Recurrent Neural Network

### Architecture:
Embedding → RNN → Dense → Sigmoid

### Steps:
- Tokenization
- Padding sequences
- Embedding layer
- RNN layer
- Training & evaluation

---

## 📌 3. Deep Learning Model (LSTM)

- Long Short-Term Memory network

### Architecture:
Embedding → Bidirectional LSTM → Dense → Dropout → Sigmoid

### Steps:
- Tokenization
- Padding sequences
- Embedding layer
- LSTM layers
- Training & evaluation

---

# 📊 Model Comparison

| Model | Type | Architecture | Accuracy | Speed | Context Understanding | Complexity |
|------|------|-------------|----------|-------|------------------------|------------|
| Logistic Regression | ML | TF-IDF → Logistic Regression | ~95–98% | ⚡ Very Fast | ❌ Low | Low |
| RNN | DL | Embedding → RNN → Dense | ~93–96% | ⚡ Medium | ⚠️ Medium | Medium |
| LSTM | DL | Embedding → BiLSTM → Dense → Dropout | ~95%+ | 🐢 Slower | ✅ High | High |

---

# 🧠 Summary

- Logistic Regression:
  - Fastest model
  - Strong baseline
  - Weak context understanding

- RNN:
  - Handles sequences
  - Medium performance
  - Weak on long dependencies

- LSTM:
  - Best performance in NLP
  - Strong context understanding
  - Slowest but most powerful

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- NLP Techniques

---

# 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
