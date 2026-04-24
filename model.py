import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# ===== Load Data =====
# الملفات موجودة داخل مجلد Dataset في المشروع
fake_df = pd.read_csv("Dataset/Fake.csv", engine="python", on_bad_lines="skip")
true_df = pd.read_csv("Dataset/True.csv", engine="python", on_bad_lines="skip")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)

# دمج العنوان مع النص
df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)
df = df[["text", "label"]]

# ===== Preprocessing =====
vocab_size = 10000
max_length = 200

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

train_pad = pad_sequences(train_seq, maxlen=max_length, padding="post")
test_pad = pad_sequences(test_seq, maxlen=max_length, padding="post")

# ===== Model =====
model = Sequential([
    Embedding(vocab_size, 64),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(24, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ===== Training =====
history = model.fit(
    train_pad, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(test_pad, y_test)
)

# ===== Plot =====
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["train", "test"])
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["train", "test"])
plt.show()

# ===== Save Model =====
model.save("fake_news_lstm_model.h5")
print("Model Saved Successfully!")