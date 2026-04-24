"""
Load .keras file directly in browser and reconstruct model in TensorFlow.js.
"""

import tensorflow as tf
import json

print("Loading model...")
model = tf.keras.models.load_model('data/sign_model_landmarks.keras', compile=False)

print("Model loaded!")
print(f"Input: {model.input_shape}")
print(f"Output: {model.output_shape}")

# Export as JSON for manual weight loading
weights_dict = {}
for layer in model.layers:
    if layer.get_weights():
        weights_dict[layer.name] = []
        for w in layer.get_weights():
            weights_dict[layer.name].append({
                'shape': list(w.shape),
                'dtype': str(w.dtype)
            })

# Save weight shapes
with open('public/model/tfjs/weight_shapes.json', 'w') as f:
    json.dump(weights_dict, f, indent=2)

print(f"Weight shapes: {weights_dict}")

# Extract weights and save as individual files
import numpy as np

for layer_name, weights in weights_dict.items():
    layer_weights = model.get_layer(layer_name).get_weights()
    for i, w in enumerate(layer_weights):
        filename = f'public/model/tfjs/{layer_name}_weight{i}.bin'
        with open(filename, 'wb') as f:
            f.write(w.astype(np.float32).tobytes())
        print(f"Saved {filename}: {w.shape}")

print("Done!")
print("Files:", sorted(['public/model/tfjs/' + f for f in os.listdir('public/model/tfjs/')]))