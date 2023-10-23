import cv2
import numpy as np

# Завантаження попередньо навченої моделі YOLO
net = cv2.dnn.readNet("datafiles/yolov3-spp.weights", "datafiles/yolov3-spp.cfg")

# Завантаження назв класів
classes = []
with open("datafiles/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Відкриття відеопотоку з машини
# cap = cv2.VideoCapture(0)
# Відкриття відеопотоку з машини з відео
cap = cv2.VideoCapture('video/car_-_2165 (540p).mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Підготовка зображення для використання в YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Відображення детекції автомобілів
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # 2 - індекс класу для автомобілів у файлі coco.names
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Отримання координат для кутів прямокутника
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Намалювати прямокутник навколо автомобіля
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
