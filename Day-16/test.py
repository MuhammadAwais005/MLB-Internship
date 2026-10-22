
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# 1. LOAD THE SAVED MODEL
# ==========================================
model = keras.models.load_model("fashion_mnist_ann.keras")
print("Trained model loaded successfully!")


image_path = r"C:\Users\92348\Desktop\MLBench\Day-16\ladies sandal.jpg"

# ==========================================
# 3. DEFINE CLASS NAMES
# ==========================================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==========================================
# 4. LOAD AND PREPROCESS THE IMAGE
# ==========================================
from PIL import ImageOps, ImageEnhance

img = load_img(
    image_path,
    color_mode="grayscale",
    target_size=(28, 28)
)

# Invert
img = ImageOps.invert(img)

# Increase contrast
img = ImageEnhance.Contrast(img).enhance(2.0)

img_array = img_to_array(img).astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# 5. PREDICT THE CLOTHING CATEGORY
# ==========================================
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print("\n--- Prediction Result ---")
print("Predicted class:", predicted_class)
print(f"Prediction score: {confidence:.2f}%")

# ==========================================
# 6. DISPLAY THE IMAGE AND PREDICTION
# ==========================================
plt.imshow(img, cmap="gray")
plt.title(
    f"Predicted: {predicted_class}\n"
    f"Score: {confidence:.2f}%"
)
plt.axis("off")
plt.show()
