# Fake News Detection 📰 (ML vs DL Comparison)

## 📌 Overview
This project compares two different approaches for Fake News Detection:

1. **Machine Learning Model (TF-IDF + Logistic Regression)**
2. **Deep Learning Model (RNN - LSTM)**

Both models aim to classify news articles as:
- Fake News (0)
- Real News (1)

The goal is to analyze the performance difference between traditional ML and Deep Learning approaches on NLP tasks.

---

## 📂 Dataset
The dataset used is the **Fake and Real News Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains:
- Fake.csv
- True.csv

---

## ⚙️ Approaches Used

### 🧠 1. Machine Learning Model
- TF-IDF Vectorization
- Logistic Regression
- Simple and fast baseline model

#### 📊 ML Workflow:
1. Text cleaning
2. TF-IDF feature extraction
3. Train/Test split
4. Logistic Regression training
5. Evaluation

---

### 🧠 2. Deep Learning Model (RNN - LSTM)
- Tokenization (Keras Tokenizer)
- Padding sequences
- LSTM-based neural network

#### 📊 DL Workflow:
1. Text preprocessing
2. Tokenization
3. Sequence padding
4. Embedding layer
5. LSTM layer
6. Dense output layer
7. Training & evaluation

---

## 🧠 Model Architectures

### ML Model:
- TF-IDF → Logistic Regression

### DL Model:
- Embedding → LSTM → Dense → Sigmoid

---

## 📊 Performance Comparison

| Model | Type | Accuracy | Speed | Complexity |
|------|------|----------|-------|------------|
| ML (Logistic Regression) | Classical ML | ~95-98% | Fast | Low |
| DL (RNN) | Deep Learning | ~95%+ | Slower | High |

---

## 🧠 Technologies Used
- Python 🐍
- Pandas 📊
- NumPy 🔢
- Scikit-learn ⚙️
- TensorFlow / Keras 🤖
- NLP Techniques

---

## 🚀 How to Run

### 🔹 Install dependencies
```bash
pip install pandas numpy scikit-learn tensorflow streamlit
