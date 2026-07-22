"""
tests jednostkowe dla Gallery.
Odpalać jako: python3 test_gallery.py
Każda funkcja testuje jeden, konkretny przypadek - jeśli któryś padnie,
"""

from src.gallery import Gallery, transform_batch_emb
import numpy as np
import torch
import torch.nn as nn

def test_1_pusta_galeria_pierwsza_osoba():
    """Pierwsza osoba w pustej galerii dostaje etykietę osoba1."""
    gallery = Gallery(max_vecs=3)
    label = gallery.match(np.array([1.0, 0.0]), margin=0.7)
    assert label == "osoba1", f"Oczekiwano 'osoba1', dostano {label}"
    print("TEST 1 PRZESZEDŁ - pusta galeria, pierwsza osoba")


def test_2_druga_inna_osoba_dostaje_nowa_etykiete():
    """Wektor ortogonalny do już zarejestrowanego to nowa osoba."""
    gallery = Gallery(max_vecs=3)
    emb_jan = np.array([1.0, 0.0])
    emb_anna = np.array([0.0, 1.0])  # ortogonalny, similarity=0

    label_jan = gallery.match(emb_jan, margin=0.7)
    label_anna = gallery.match(emb_anna, margin=0.7)

    assert label_jan == "osoba1"
    assert label_anna == "osoba2", f"Oczekiwano 'osoba2', dostano {label_anna}"
    print("TEST 2 PRZESZEDŁ - druga, inna osoba dostaje nową etykietę")


def test_3_ta_sama_osoba_rozpoznana_ponownie():
    """Podobny, ale nie identyczny wektor tej samej osoby -> ta sama etykieta, dopisany do deque."""
    gallery = Gallery(max_vecs=3)
    emb_jan_1 = np.array([1.0, 0.0])
    emb_jan_2 = np.array([0.9, 0.1])  # podobny, "wrócił po chwili"

    label_1 = gallery.match(emb_jan_1, margin=0.7)
    label_2 = gallery.match(emb_jan_2, margin=0.7)

    assert label_1 == "osoba1"
    assert label_2 == "osoba1", f"Oczekiwano ponownie 'osoba1', dostano {label_2}"
    assert len(gallery.known_emb["osoba1"]) == 2, "Drugi wektor powinien dopisać się do deque"
    print("TEST 3 PRZESZEDŁ - ta sama osoba rozpoznana ponownie, dopisana do deque")


def test_4_najlepsze_dopasowanie_wsrod_wielu_osob():
    """Argmax poprawnie wybiera najbliższą osobę, nie pierwszą z brzegu."""
    gallery = Gallery(max_vecs=3)
    gallery.match(np.array([1.0, 0.0]), margin=0.7)   # osoba1 (Jan)
    gallery.match(np.array([0.0, 1.0]), margin=0.7)   # osoba2 (Anna)

    label = gallery.match(np.array([0.0, 0.95]), margin=0.7)  # bliski Annie
    assert label == "osoba2", f"Oczekiwano 'osoba2', dostano {label}"
    print("TEST 4 PRZESZEDŁ - najlepsze dopasowanie wśród wielu osób")


def test_5_przepelnienie_deque():
    """Po przekroczeniu max_vecs najstarszy wektor znika automatycznie."""
    gallery = Gallery(max_vecs=2)
    gallery.match(np.array([1.0, 0.0]), margin=0.7)     # osoba1, wektor A
    gallery.match(np.array([0.95, 0.05]), margin=0.7)   # osoba1, wektor B
    gallery.match(np.array([0.9, 0.1]), margin=0.7)     # osoba1, wektor C -> A powinno zniknąć

    deque_osoba1 = gallery.known_emb["osoba1"]
    assert len(deque_osoba1) == 2, f"Oczekiwano max 2 elementów, jest {len(deque_osoba1)}"
    print("TEST 5 PRZESZEDŁ - przepełnienie deque, najstarszy wektor wypchnięty")


def test_6_renormalizacja_po_usrednieniu():
    """Uśredniony wektor musi mieć z powrotem długość ~1 (do poprawnego cosine similarity)."""
    gallery = Gallery(max_vecs=3)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.7071, 0.7071])  # 45 stopni od v1, oba jednostkowe

    gallery.match(v1, margin=0.5)
    gallery.match(v2, margin=0.5)  # cos(45°)=0.71 > 0.5, więc dopasuje się do v1

    # replikujemy dokładnie te same kroki co wewnątrz match(), żeby sprawdzić
    # co faktycznie trafia do similarities (nie liczymy średniej "z zewnątrz")
    emb_list = np.array(list(gallery.known_emb.values()))
    emb_matrix_pre = np.array(list(np.mean(i, axis=0) for i in emb_list))
    emb_matrix = np.array(list(row / (np.linalg.norm(row) + 1e-6) for row in emb_matrix_pre))

    dlugosc_przed = np.linalg.norm(emb_matrix_pre[0])
    dlugosc_po = np.linalg.norm(emb_matrix[0])

    assert dlugosc_przed < 0.99, "Kontrolnie: przed renormalizacją długość powinna być < 1"
    assert abs(dlugosc_po - 1.0) < 1e-4, f"Oczekiwano długości ~1.0, dostano {dlugosc_po}"
    print(f"TEST 6 PRZESZEDŁ - renormalizacja: {dlugosc_przed:.4f} -> {dlugosc_po:.4f}")

class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data

def test_7_batch_emb():
    data = [
        [1.0, 0.0],
        [0.7071, 0.7071]
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IdentityModel().to(device)
    model.eval()

    emb = transform_batch_emb(data, device, model)

    print("wynik transform_batch_emb:", emb)
    print("dlugosc wyniku:", np.linalg.norm(emb))

    oczekiwany_kierunek = np.array([0.85355, 0.35355])
    oczekiwany_kierunek_znormalizowany = oczekiwany_kierunek / np.linalg.norm(oczekiwany_kierunek)

    assert np.linalg.norm(emb) - 1.0 < 1e-4, f"Oczekiwano dlugosci ~1.0, dostano {np.linalg.norm(emb)}"
    assert np.allclose(emb, oczekiwany_kierunek_znormalizowany,
                       atol=1e-3), f"Kierunek sie nie zgadza: {emb} vs {oczekiwany_kierunek_znormalizowany}"
    print("TEST PRZESZEDL - transform_batch_emb z IdentityModel dziala poprawnie")


if __name__ == "__main__":
    test_1_pusta_galeria_pierwsza_osoba()
    test_2_druga_inna_osoba_dostaje_nowa_etykiete()
    test_3_ta_sama_osoba_rozpoznana_ponownie()
    test_4_najlepsze_dopasowanie_wsrod_wielu_osob()
    test_5_przepelnienie_deque()
    test_6_renormalizacja_po_usrednieniu()
    test_7_batch_emb()
    print()
    print("WSZYSTKIE TESTY PRZESZŁY")