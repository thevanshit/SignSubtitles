import tensorflow as tf
import os
import json

print(f"TensorFlow version: {tf.__version__}")

print("\nCreating model with same architecture...")
n_timesteps = 30
n_features = 225
n_classes = 9

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created. Input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

model.summary()

print("\nCreating TensorFlow.js format files...")

output_dir = 'public/model'
os.makedirs(output_dir, exist_ok=True)

model.save(os.path.join(output_dir, 'model_tf.keras'))

print(f"\nModel saved to {output_dir}/model_tf.keras")
print("\nDone!")