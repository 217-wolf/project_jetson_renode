#!/usr/bin/env python3
import cv2
import subprocess
import sys

def list_video_devices():
    print("Dostępne urządzenia video:")
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
        print(result.stdout if result.stdout else "Brak urządzeń lub v4l2-ctl niedostępny")
    except FileNotFoundError:
        print("v4l2-ctl nie zainstalowane. Instaluj: sudo apt install v4l-utils")
    # Alternatywnie ls /dev/video*
    import glob
    devices = glob.glob('/dev/video*')
    print(f"Znalezione /dev/video*: {devices}")

def test_camera(index, backend):
    if backend == 'v4l2':
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    elif backend == 'gstreamer':
        # Prosty pipeline dla testu
        pipeline = f"v4l2src device=/dev/video{index} ! videoconvert ! video/x-raw,format=BGR ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"  Nie można otworzyć kamery {index} backend {backend}")
        return False
    
    # Pobierz właściwości
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Udało się otworzyć: {width}x{height} @ {fps}fps")
    
    # Spróbuj odczytać klatkę
    ret, frame = cap.read()
    if ret:
        print(f"  Odczytano klatkę o kształcie {frame.shape}")
        cap.release()
        return True
    else:
        print("  Nie udało się odczytać klatki")
        cap.release()
        return False

def main():
    print("Diagnostyka kamery na Jetson Orin Nano")
    list_video_devices()
    
    # Testuj indeksy 0,1,2 z różnymi backendami
    for index in range(3):
        print(f"\nTestowanie indeksu {index}:")
        # Najpierw domyślny
        if test_camera(index, 'default'):
            print(f"Znaleziono działającą kamerę na indeksie {index} (default). Uruchamianie podglądu...")
            run_viewer(index, 'default')
            return
        # V4L2
        if test_camera(index, 'v4l2'):
            print(f"Znaleziono działającą kamerę na indeksie {index} (V4L2). Uruchamianie podglądu...")
            run_viewer(index, 'v4l2')
            return
        # GStreamer
        if test_camera(index, 'gstreamer'):
            print(f"Znaleziono działającą kamerę na indeksie {index} (GStreamer). Uruchamianie podglądu...")
            run_viewer(index, 'gstreamer')
            return
    
    print("Nie znaleziono żadnej działającej kamery. Sprawdź połączenie i uprawnienia.")
    print("Możesz spróbować: sudo usermod -a -G video $USER (potem wyloguj/zaloguj)")

def run_viewer(index, backend):
    if backend == 'v4l2':
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    elif backend == 'gstreamer':
        # Nieco bardziej niezawodny pipeline z autodetekcją formatu
        pipeline = f"v4l2src device=/dev/video{index} ! videoconvert ! video/x-raw,format=BGR ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print("Błąd ponownego otwarcia kamery")
        return
    
    # Ustaw niższą rozdzielczość dla wydajności
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Wyświetlanie obrazu. Naciśnij 'q' aby zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd odczytu klatki")
            break
        cv2.imshow("Kamera USB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()