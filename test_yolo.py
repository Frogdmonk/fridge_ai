import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

video_path = "fridge.mp4"

cap = cv2.VideoCapture(video_path)

detected_items = set()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % 30 == 0:
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < 0.50:
                    continue

                class_id = int(box.cls[0])
                name = model.names[class_id]

                detected_items.add(name)

                print(f"Detected: {name} ({confidence:.2f})")

    frame_count += 1

cap.release()

print("\n==========================")
print("FINAL DETECTED OBJECTS")
print("==========================")

for item in detected_items:
    print("-", item)
