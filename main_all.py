import cv2
import tensorflow as tf
import numpy as np

# Завантаження попередньо навченої моделі
model = tf.keras.applications.MobileNetV2(weights='imagenet')


# Функція для обробки зображення
def process_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# Відкриваємо відеофайл
# cap = cv2.VideoCapture('.mp4') # вкажи свій шлях на відео

# Відкриття відеопотоку
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = process_image(frame)
    predictions = model.predict(processed_frame)
    decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=2)

    # Виведення результатів
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        cv2.putText(frame, '{} - {:.2f}%'.format(label, score * 100), (10, (i + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
