"""
Train LSTM model with legacy optimizer for compatibility.
"""

import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.optimizers import legacy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

PHRASE_MAP = {
    'HELLO': 0, 'HRU': 1, 'FINE': 2, 'YES': 3, 'NO': 4,
    'HELP': 5, 'THANK': 6, 'PLEASE': 7, 'SLOW': 8, 'NICE': 9,
}

print("=" * 50)
print("Training LSTM on Hand Landmarks")
print("=" * 50)

print("\n[1] Loading landmark data...")
X = np.load('data/X_landmarks.npy')
y = np.load('data/y_landmarks.npy')

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

n_timesteps = X.shape[1]
n_features = X.shape[2]
n_classes = len(PHRASE_MAP)

print("\n[2] Preprocessing...")
X_flat = X.reshape(-1, n_features)
mean = X_flat.mean(axis=0)
std = X_flat.std(axis=0) + 1e-8
X_norm = (X_flat - mean) / std
X = X_norm.reshape(X.shape)

scaler = {'mean': mean.tolist(), 'std': std.tolist()}
with open('data/scaler_landmarks.json', 'w') as f:
    json.dump(scaler, f)
print("Saved scaler_landmarks.json")

print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

print("\n[4] Building model...")
model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=legacy.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

print("\n[5] Training...")
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2)

print("\n[6] Evaluating...")
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
class_names = list(PHRASE_MAP.keys())

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n[7] Saving model...")
model.save('data/sign_model_landmarks.keras')
print("Saved: data/sign_model_landmarks.keras")

metadata = {
    'class_names': class_names,
    'input_shape': [None, n_timesteps, n_features],
    'num_classes': n_classes,
    'feature_type': 'hand_landmarks',
    'n_landmarks': 21
}
with open('public/model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Saved: public/model/metadata.json")

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)