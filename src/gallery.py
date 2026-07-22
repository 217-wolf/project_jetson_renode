import numpy as np
from collections import deque, defaultdict
import torch

class Gallery:
    def __init__(self, max_vecs):
        self.next_person = 1
        self.max_vecs = max_vecs
        self.known_emb = defaultdict(lambda: deque(maxlen=self.max_vecs)) #galeria to dziennik o pojemnosci max_vecs na key -> deque nam to gwarantuje

    def match(self, new_emb, margin):
        if not self.known_emb:
            id = f"osoba{self.next_person}"
            self.next_person += 1
            self.known_emb[id].append(new_emb)
            return id #zwracamy id:str bo bedziemy to wyswietlac na ekranie, dziennik aktualizuje sie i tak

        emb_list = np.array(list(self.known_emb.values()))
        emb_matrix_pre = np.array(list((np.mean(i, axis=0) for i in emb_list))) #jezeli maxlen > 1
        emb_matrix = np.array(list(row / (scale := np.linalg.norm(row) + 1e-6) for row in emb_matrix_pre)) #usrednienie wyzej powoduje, ze wektory nie są juz znormalizowane, naprawiamy

        similarites = emb_matrix @ new_emb

        # co jezeli jeden wektor ma 0.7, a inny 0.69 dla roznych id???? Czy to czasami nie ta sama osoba, nie nie warto sprawdzic wtedy obu id z galleri czy nie sa podobne do siebie i w razie czego je połączyć?
        best_idx = np.argmax(similarites)
        best_similarity = similarites[best_idx]

        if best_similarity <= margin:
            id = f"osoba{self.next_person}"
            self.next_person += 1
            self.known_emb[id].append(new_emb)
            return id
        else:
            emb_ids = np.array(list(self.known_emb.keys()))
            id = emb_ids[best_idx]
            self.known_emb[id].append(new_emb)
            return id


def transform_batch_emb(data, device, model):
    """
    Uśrednienie robimy po przejściu batcha klatek przez sieć.
    Uśredniony wektor przed wejściem do sieci zwróci nam nowy sztuczny szkielet, który nigdy nie miał miejsca.

    Wymaga przed wywołaniem model.eval() -ta funkcja tego nie robi, robimy to przed pęlą while cap.isOpened(): w live_pipeline.py
    """

    batch = np.array(list(data))
    batch_torch = torch.from_numpy(batch).float().to(device)

    with torch.no_grad():
        emb_batch_pre = model(batch_torch)
        emb_batch_pre  = emb_batch_pre.detach().cpu().numpy()
        emb_pre = np.mean(emb_batch_pre, axis=0)
        emb = emb_pre / (np.linalg.norm(emb_pre) +1e-6)

    return emb