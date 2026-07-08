import socket
import json
import time
# Połącz się raz na początku
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('adres-trackera', 5555))

# W pętli po każdym oknie czasowym:
msg = {
    "jetson_id": "jetson1",
    "timestamp": time.time(),
    "detections": [{
        "local_id": track.local_id,
        "class": track.class_name,
        "bbox": list(track.bbox),
        "embedding": track.embedding.tolist(),
        "confidence": track.conf
    } for track in local_tracks] #jest to lista w ktorej maja byc szukane obiekty 
    #a wiec potrzeba jej deklaracji w glownym kodzie oraz wstawienia tam tych wzorcow suzkanych
}
sock.sendall((json.dumps(msg) + "\n").encode())