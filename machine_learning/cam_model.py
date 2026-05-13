#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

def main():
    # 1. Wczytaj model YOLO (najlżejszy - nano, szybki na Jetson)
    model = YOLO('yolov8n.pt')   # możesz też 'yolov8s.pt' (większy, dokładniejszy)
    # Wymuszenie użycia GPU (CUDA) jeśli dostępne
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        model.to('cuda')
        print("YOLO działa na GPU (CUDA)")
    else:
        print("YOLO działa na CPU")

    # 2. Otwórz kamerę (indeks 0 – pierwsza kamera USB)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    # Opcjonalnie zmniejsz rozdzielczość dla lepszej wydajności
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Wyświetlanie obrazu z detekcjami YOLO. Naciśnij 'q' aby zakończyć.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wykonaj detekcję YOLO na bieżącej klatce
        results = model(frame, verbose=False)   # verbose=False aby nie zaśmiecać konsoli

        # Narysuj bounding boxy, etykiety i pewność na obrazie
        annotated_frame = results[0].plot()     # results[0] – pierwszy obraz (batch size 1)

        # Wyświetl
        cv2.imshow("YOLO + Kamera Microsoft", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()