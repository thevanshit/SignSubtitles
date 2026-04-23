import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import json

print("Loading data...")
data = np.load('train_data.npy', allow_pickle=True)

X = np.array([item['sequence'] for item in data])
y = np.array([item['label'] for item in data])

print(f"Original shape: X={X.shape}, y={y.shape}")

# Save class names for frontend
classes = sorted(list(np.unique(y)))
metadata = {
    "class_names": classes
}
with open('public/model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Classes: {classes}")

# Normalize using StandardScaler (feature-wise normalization)
# Reshape to (samples * frames, features) for fitting
X_reshaped = X.reshape(-1, X.shape[-1])
print(f"Reshaped for scaler: {X_reshaped.shape}")

scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_reshaped)

# Reshape back to (samples, frames, features)
X = X_scaled_flat.reshape(X.shape)
print(f"Normalized shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save scaler and encoder for inference
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Saved scaler.pkl and label_encoder.pkl")

# Train/test split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Model architecture
n_timesteps = X.shape[1]
n_features = X.shape[2]
n_classes = len(classes)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
]

print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc:.2%}")

# Save Keras model
model.save('sign_model.keras')
print("Saved sign_model.keras")

print("\nTraining complete!")