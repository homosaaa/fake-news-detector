# Fake News Detector 📰

## 📌 Overview
This project is a Machine Learning-based Fake News Detection system that classifies news articles as either **Fake (0)** or **Real (1)** using Natural Language Processing (NLP) techniques.

The model is trained using a dataset of real and fake news articles and uses **TF-IDF Vectorization** with **Logistic Regression** for classification.

A simple **Streamlit web app** is built to allow real-time predictions.

---

## 📂 Dataset
The dataset used in this project is the **Fake and Real News Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

It contains:
- Fake.csv → Fake news articles
- True.csv → Real news articles

Each article includes:
- Title
- Text
- Subject
- Date
- Label (0 = Fake, 1 = Real)

---

## ⚙️ Workflow
1. Load dataset
2. Assign labels (Fake = 0, Real = 1)
3. Merge datasets
4. Clean text data
5. Split into training and testing sets
6. Convert text into numerical features using TF-IDF
7. Train Logistic Regression model
8. Evaluate model performance
9. Deploy using Streamlit

---

## 🧠 Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- NLP (TF-IDF)

---

## 📊 Model Performance
The model achieved high accuracy (~98%) on the test dataset, showing strong performance in distinguishing between fake and real news.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit pandas scikit-learn
