import cv2

cap = cv2.VideoCapture("fridge.mp4")

print("Opened:", cap.isOpened())
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
