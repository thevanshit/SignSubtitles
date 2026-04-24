"""
Manually convert Keras model to TFJS format.
Extracts weights and creates model.json manually.
"""

import os
import json
import numpy as np

print("Loading model...")
model_path = 'data/sign_model_landmarks.keras'

# Load model weights manually
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = tf.keras.models.load_model(model_path, compile=False)

print("Model loaded!")
print(f"Input: {model.input_shape}")
print(f"Output: {model.output_shape}")

# Extract weights
weights = []
for layer in model.layers:
    if layer.get_weights():
        layer_weights = []
        for w in layer.get_weights():
            layer_weights.append(w.tolist())
        weights.append({
            'layer_name': layer.name,
            'weights': layer_weights
        })
        print(f"Layer {layer.name}: {len(layer.get_weights())} weight tensors")

# Create model.json (TFJS format)
model_json = {
    "format": "layers-model",
    "generatedBy": "keras v2.15.0",
    "convertedBy": "tensorflowjs converted",
    "modelTopology": {
        "keras_version": "2.15.0",
        "backend": "tensorflow"
    },
    "weightsManifest": [
        {
            "paths": ["weights.bin"],
            "weights": []
        }
    ]
}

# Add topology
config = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": []
    },
    "keras_version": "2.15.0"
}

for i, layer in enumerate(model.layers):
    layer_config = {
        "class_name": layer.__class__.__name__,
        "config": layer.get_config(),
        "name": layer.name
    }
    config["config"]["layers"].append(layer_config)

model_json["modelTopology"] = config

# Save files
os.makedirs('public/model/tfjs', exist_ok=True)

with open('public/model/tfjs/model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("Saved model.json")

# Flatten and save weights
flat_weights = []
for layer in model.layers:
    for w in layer.get_weights():
        flat_weights.extend(w.flatten())

weights_array = np.array(flat_weights, dtype=np.float32)

with open('public/model/tfjs/weights.bin', 'wb') as f:
    f.write(weights_array.tobytes())

print(f"Saved weights.bin ({len(weights_array)} weights)")

# Also save metadata
import pickle
with open('data/label_encoder_landmarks.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

metadata = {
    "class_names": list(label_encoder.keys()),
    "input_shape": [None, 30, 63],
    "num_classes": len(label_encoder),
    "feature_type": "hand_landmarks"
}

with open('public/model/tfjs/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Done!")
print("Files:", os.listdir('public/model/tfjs'))