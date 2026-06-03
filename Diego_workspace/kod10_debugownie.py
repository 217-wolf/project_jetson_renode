# check_cameras.py
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        status = "OK" if ret else "Błąd odczytu"
        print(f"Kamera {i}: otwarta – {status}")
    else:
        print(f"Kamera {i}: niedostępna")