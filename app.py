import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory

def predict_image(model, image_path, class_names):
    img = keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predictions = predictions[0]
    
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence      = predictions[predicted_index] * 100

    print(f"\n>>> Prediction for: {image_path}")
    print(f"   - Predicted class : {predicted_class}")
    print(f"   - Confidence      : {confidence:.1f}%")
    print(f"   - All probabilities:")
    for name, prob in zip(class_names, predictions):
        bar = "█" * int(prob * 20)
        print(f"      {name:<15} {prob*100:5.1f}%  {bar}")

    return predicted_class, confidence

if __name__ == "__main__":
    IMG_HEIGHT   = 128
    IMG_WIDTH    = 128
    BATCH_SIZE   = 32

    val_dir = r"datasets\val"
    val_dataset = image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        seed=42
    )

    class_names = val_dataset.class_names

    image_path = r"images\cardboard-trash.jpg"

    model = keras.models.load_model(r'models\best_model.keras')
    # model.summary()

    predict_image(model, image_path, class_names)
