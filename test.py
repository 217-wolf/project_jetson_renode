import cv2
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

cap = cv2.VideoCapture(0)

print("'q', aby wyjsc")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    #class=0 to person
    #stream=True daje lepsza wydajnosc przy wideo
    results = model.track(frame, persist=True, stream=True, classes=0, tracker="botsort.yaml")

    for r in results:
        annotated_frame = r.plot() #rysuje ramki i szkielet

        if r.keypoints is not None:
            # keypoints.xyn to znormalizowane (0-1) współrzędne [x, y] stawów
            # Są idealne do analizy postury (Gait Analysis) niezależnie od rozdzielczości
            points = r.keypoints.xyn.cpu().numpy()

            for person_idx, kpts in enumerate(points):
                if len(kpts) > 0:
                    # kpts[0] to nos, kpts[15] i [16] to kostki itd.
                    # To te dane będziecie porównywać z datasetem SURREAL
                    pass

    cv2.imshow("YOLOv11 Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()