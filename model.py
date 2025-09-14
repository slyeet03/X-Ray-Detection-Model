import tensorflow as tf
import numpy as np
import os
import json

# configuration
MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
LABELS = ["Pneumothorax", "Pneumonia", "Edema", "Pleural Effusion", "Consolidation", "Cardiomegaly", "Atelectasis"]

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# img processing
def load_process_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# predict
def predict(path):
    image = load_process_image(path)
    preds = model.predict(image, verbose=0)[0]
    result = {label: float(prob) for label, prob in zip(LABELS, preds)}
    return result

if __name__ == "__main__":
    folder = "test_images"
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                img_path = os.path.join(folder, fname)
                result = predict(img_path)

                # save JSON
                json_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.json")
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=4)

                # find top prediction
                top_label = max(result, key=result.get)
                top_prob = result[top_label] * 100

                # print top result
                print(f"\nPrediction for: {fname}")
                print(f"Top: {top_label} ({top_prob:.2f}%)")
