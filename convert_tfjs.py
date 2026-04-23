import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import tensorflow as tf

model_path = 'sign_model.keras'
output_path = 'public/model'

os.makedirs(output_path, exist_ok=True)

print("Loading Keras model...")
model = tf.keras.models.load_model(model_path)

print("Converting to TensorFlow.js format...")
tf.keras.backend.set_learning_phase(0)
export_path = os.path.join(output_path, 'tfjs_model')
model.save(export_path)

print(f"Model saved to {export_path}")
print("Contents:", os.listdir(export_path))

print("\nUpdating metadata.json...")
metadata = {
    "class_names": ["HELLO", "HELP", "HOW ARE YOU", "I AM FINE", "NICE TO MEET YOU", "NO", "PLEASE REPEAT", "SLOW DOWN", "THANK YOU", "YES"]
}
with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Done!")