# =========================
# 1) Import Libraries
# =========================
import pandas as pd
import numpy as np


uploaded = files.upload()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split

# =========================
# 2) Load Data
# =========================

# ===== Load Data =====

df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')
# Add labels
df_fake['label'] = 0
df_true['label'] = 1

# Merge
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# =========================
# 3) Choose Text Column
# =========================
# لو عندك column اسمها text أو title
print(df.columns)

X = df['text']   # غيرها لو اسم العمود مختلف
y = df['label']

# =========================
# 4) Tokenization
# =========================
vocab_size = 10000
max_len = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# =========================
# 5) Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    padded, y, test_size=0.2, random_state=42
)

# =========================
# 6) Build RNN Model (LSTM)
# =========================
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================
# 7) Train Model
# =========================
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# =========================
# 8) Evaluate
# =========================
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# =========================
# 9) Test Prediction
# =========================
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0][0]
    return "True News ✅" if pred > 0.5 else "Fake News ❌"

print(predict_text("Breaking news: something happened today"))
