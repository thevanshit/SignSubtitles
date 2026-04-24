"""
Rebuild model without training and export for TFJS.
"""

import os
import json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.optimizers import legacy as legacy_opt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

PHRASE_MAP = {
    'HELLO': 0, 'HRU': 1, 'FINE': 2, 'YES': 3, 'NO': 4,
    'HELP': 5, 'THANK': 6, 'PLEASE': 7, 'SLOW': 8, 'NICE': 9,
}

n_timesteps = 30
n_features = 63
n_classes = len(PHRASE_MAP)

model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=legacy_opt.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("Loading and processing data...")
X = np.load('data/X_landmarks.npy')
y = np.load('data/y_landmarks.npy')

X_flat = X.reshape(-1, n_features)
mean = X_flat.mean(axis=0)
std = X_flat.std(axis=0) + 1e-8
X_norm = (X_flat - mean) / std
X = X_norm.reshape(X.shape)

print("Training quick...")
model.fit(X, y, epochs=30, batch_size=16, verbose=0)

print("Saving as SavedModel...")
os.makedirs('public/model', exist_ok=True)
model.save('public/model/keras_model')

print("Converting to TFJS...")
import subprocess
result = subprocess.run(
    ['tensorflowjs_converter', '--input_format', 'keras', 
     '--output_format', 'tfjs_layers_model',
     'public/model/keras_model', 'public/model/tfjs_model'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("TFJS converter not available, using Keras format")
    print("Saved as Keras model - will load in browser instead")

print("\nUpdating metadata...")
metadata = {
    'class_names': list(PHRASE_MAP.keys()),
    'input_shape': [None, n_timesteps, n_features],
    'num_classes': n_classes,
    'feature_type': 'hand_landmarks',
    'scaler_mean': mean.tolist(),
    'scaler_std': std.tolist()
}
with open('public/model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Done!")
print("Files:", os.listdir('public/model'))