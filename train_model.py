import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("Loading data...")
with open('train_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array([d['sequence'] for d in data])
y_labels = [d['label'] for d in data]

print(f"Sequences: {X.shape}, Labels: {len(y_labels)}")

le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
print(f"Classes: {list(le.classes_)}")

y_onehot = tf.keras.utils.to_categorical(y_encoded)
print(f"Label shape: {y_onehot.shape}")

X_flat = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)
X_train_flat, X_val_flat, y_train, y_val = train_test_split(
    X_flat_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

n_timesteps = X.shape[1]
n_features = X.shape[2]
n_classes = len(le.classes_)

X_train = X_train_flat.reshape(-1, n_timesteps, n_features)
X_val = X_val_flat.reshape(-1, n_timesteps, n_features)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy: {val_acc:.2%}")

model.save('sign_lstm_model.keras')
print("Saved sign_lstm_model.keras")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved label_encoder.pkl and scaler.pkl")

print("\nTraining complete!")
