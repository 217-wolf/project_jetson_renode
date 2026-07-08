#!/usr/bin/env python3
"""
Globalny tracker wielokamerowy (wielo-Jetsonowy) z ReID.
Odbiera dane z wielu Jetsonów przez TCP, kojarzy embeddingi i przydziela globalne ID.
Może działać na wydzielonym węźle (centralny serwer) lub na jednym z Jetsonów.
"""

import socket
import threading
import json
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------- KONFIGURACJA ----------------------------------------
HOST = '0.0.0.0'
PORT = 5555
MATCHING_THRESHOLD = 0.75          # próg podobieństwa cosinusowego
MAX_AGE = 5.0                      # maks. czas życia ścieżki [s] bez aktualizacji
SYNC_WINDOW = 0.2                  # okno synchronizacji [s] – zbieramy dane przez ten czas
# -----------------------------------------------------------------------------

class GlobalTracker:
    def __init__(self):
        self.tracks = {}            # global_id -> track_info
        self.next_id = 0
        self.lock = threading.Lock()
        # Bufor wiadomości z ostatnich SYNC_WINDOW sekund
        self.buffer = []            # list of (timestamp, jetson_id, detections_list)

    def _new_global_id(self):
        self.next_id += 1
        return self.next_id

    def add_jetson_data(self, jetson_id, timestamp, detections):
        """Dodaj paczkę detekcji do bufora (wątek serwera)."""
        with self.lock:
            self.buffer.append((timestamp, jetson_id, detections))

    def process_window(self):
        """
        Wyciągnij z bufora wszystkie paczki, które są w aktualnym oknie synchronizacji,
        przeprowadź skojarzenie i zwróć mapowanie (jetson_id -> lista {local_id: global_id}).
        """
        with self.lock:
            now = time.time()
            # Pobierz paczki nie starsze niż SYNC_WINDOW
            relevant = [item for item in self.buffer if now - item[0] <= SYNC_WINDOW]
            self.buffer = [item for item in self.buffer if now - item[0] > SYNC_WINDOW]
            # Odrzuć też przestarzałe pakiety
            # (W normalnym działaniu bufor będzie czyszczony na bieżąco)

        if not relevant:
            return {}

        # Zbierz wszystkie detekcje z tego okna
        all_detections = []
        for ts, jid, dets in relevant:
            for d in dets:
                all_detections.append({
                    'jetson_id': jid,
                    'local_id': d['local_id'],
                    'class': d['class'],
                    'embedding': np.array(d['embedding']),
                    'confidence': d.get('confidence', 1.0)
                })

        # Pobierz aktywne globalne ścieżki
        with self.lock:
            active_tracks = {
                gid: track for gid, track in self.tracks.items()
                if now - track['last_seen'] <= MAX_AGE
            }

        if not active_tracks:
            # Brak istniejących ścieżek – wszystkie nowe detekcje dostają nowe ID
            assignments = {}
            with self.lock:
                for det in all_detections:
                    gid = self._new_global_id()
                    self.tracks[gid] = {
                        'class': det['class'],
                        'embedding': det['embedding'],
                        'last_seen': now,
                        'sources': {det['jetson_id']}
                    }
                    assignments.setdefault(det['jetson_id'], []).append(
                        {'local_id': det['local_id'], 'global_id': gid}
                    )
            return assignments

        # Macierz kosztów: wiersze = globalne ID, kolumny = nowe detekcje
        track_ids = list(active_tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(all_detections)))

        for i, gid in enumerate(track_ids):
            track_emb = active_tracks[gid]['embedding']
            track_class = active_tracks[gid]['class']
            for j, det in enumerate(all_detections):
                if det['class'] != track_class:
                    cost_matrix[i, j] = 1e9  # duży koszt dla niezgodnych klas
                else:
                    # odległość cosinusowa (1 - podobieństwo)
                    sim = np.dot(track_emb, det['embedding']) / (
                        np.linalg.norm(track_emb) * np.linalg.norm(det['embedding'])
                    )
                    cost_matrix[i, j] = 1.0 - sim

        # Rozwiązanie problemu przypisania (algorytm węgierski)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Przetwarzanie przypisań
        assignments = defaultdict(list)
        matched_det_indices = set()
        matched_track_indices = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - MATCHING_THRESHOLD):
                gid = track_ids[r]
                det = all_detections[c]
                with self.lock:
                    # Aktualizacja ścieżki: wygładzanie embeddingu (średnia ruchoma)
                    alpha = 0.8
                    self.tracks[gid]['embedding'] = (
                        alpha * self.tracks[gid]['embedding'] +
                        (1 - alpha) * det['embedding']
                    )
                    self.tracks[gid]['last_seen'] = now
                    self.tracks[gid]['sources'].add(det['jetson_id'])
                assignments[det['jetson_id']].append({
                    'local_id': det['local_id'],
                    'global_id': gid
                })
                matched_det_indices.add(c)
                matched_track_indices.add(r)

        # Dodaj nowe ścieżki dla nieprzypisanych detekcji
        for j, det in enumerate(all_detections):
            if j not in matched_det_indices:
                gid = self._new_global_id()
                with self.lock:
                    self.tracks[gid] = {
                        'class': det['class'],
                        'embedding': det['embedding'],
                        'last_seen': now,
                        'sources': {det['jetson_id']}
                    }
                assignments[det['jetson_id']].append({
                    'local_id': det['local_id'],
                    'global_id': gid
                })

        # (opcjonalnie) usuń stare ścieżki spoza max_age – następnym razem będą ignorowane
        # tutaj nie usuwamy na stałe, można dodać osobny wątek czyszczący

        return dict(assignments)

# ----------------------- OBSŁUGA SIECI (serwer TCP) ---------------------------
class TrackerServer:
    def __init__(self, host, port):
        self.tracker = GlobalTracker()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.clients = {}   # conn -> jetson_id (ustawiane po pierwszej wiadomości)
        self.running = True
        # Uruchom wątek okresowego przetwarzania okna
        self.process_thread = threading.Thread(target=self._periodic_process, daemon=True)
        self.process_thread.start()

    def _periodic_process(self):
        """Co SYNC_WINDOW sekund wykonaj skojarzenie i wyślij wyniki."""
        while self.running:
            time.sleep(SYNC_WINDOW)
            assignments = self.tracker.process_window()
            if assignments:
                self.send_assignments(assignments)

    def send_assignments(self, assignments):
        """Wyślij przypisania do odpowiednich klientów (Jetsonów)."""
        with threading.Lock():
            for conn, jid in self.clients.items():
                if jid in assignments:
                    msg = json.dumps({"assignments": {jid: assignments[jid]}}) + '\n'
                    try:
                        conn.sendall(msg.encode())
                    except:
                        pass  # klient może być rozłączony, obsłużymy przy odczycie

    def handle_client(self, conn, addr):
        print(f"Nowe połączenie: {addr}")
        jetson_id = None
        buffer = ""
        while self.running:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                        jetson_id = msg.get('jetson_id', str(addr))
                        ts = msg.get('timestamp', time.time())
                        detections = msg.get('detections', [])
                        self.tracker.add_jetson_data(jetson_id, ts, detections)
                        # Zapamiętaj mapowanie
                        if jetson_id:
                            self.clients[conn] = jetson_id
                    except json.JSONDecodeError:
                        print(f"Błędny JSON od {addr}: {line[:100]}")
            except (ConnectionResetError, BrokenPipeError):
                break
        print(f"Rozłączono: {addr} (jetson: {jetson_id})")
        if conn in self.clients:
            del self.clients[conn]
        conn.close()

    def start(self):
        print(f"Global tracker nasłuchuje na {HOST}:{PORT}")
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
            except KeyboardInterrupt:
                self.running = False
                break
        self.server_socket.close()

    def stop(self):
        self.running = False

# ----------------------- MAIN -------------------------------------------------
if __name__ == "__main__":
    server = TrackerServer(HOST, PORT)
    try:
        server.start()
    except KeyboardInterrupt:
        print("Zatrzymywanie globalnego trackera")
        server.stop()