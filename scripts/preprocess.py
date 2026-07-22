"""
Konwertuje casia-b_pose_*.csv na słownik {subject_id: [wektor_34d, wektor_34d, ...]}
i zapisuje do .npz (dużo szybsze do wczytania niż CSV przy każdym treningu).

Kolejność kolumn w CSV odpowiada dokładnie kolejności COCO-17,
nos, lewe_oko, prawe_oko, lewe_ucho, prawe_ucho,
lewy_bark, prawy_bark, lewy_lokiec, prawy_lokiec,
lewy_nadgarstek, prawy_nadgarstek, lewe_biodro, prawe_biodro,
lewe_kolano, prawe_kolano, lewa_kostka, prawa_kostka
"""

import numpy as np
import csv
import re
from collections import defaultdict
from src.skeleton_utils import normalize_skeleton



# wyciąga ID osoby z nazwy pliku typu ./001-bg-01-000/000001.jpg -> "001"
NAME_RE = re.compile(r"\./(\d+)-")
MIN_CONF = 0.5  # klatki z gorszą średnią pewnością odrzucamy


def process_csv(path, min_conf=MIN_CONF):
    subject_data = defaultdict(list)
    skipped_low_conf = 0
    skipped_bad_hip = 0
    total = 0

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            total += 1
            name = row[0]
            match = NAME_RE.search(name)
            if not match:
                continue
            subject_id = match.group(1)

            values = np.array(row[1:], dtype=np.float32)
            # co 3 liczby: x, y, conf -> 17 punktów
            values = values.reshape(17, 3)
            kpts = values[:, :2]
            confs = values[:, 2]

            if confs.mean() < min_conf:
                skipped_low_conf += 1
                continue

            # biodra muszą być wykryte sensownie, bo na nich opiera się normalizacja
            if confs[11] < 0.3 or confs[12] < 0.3:
                skipped_bad_hip += 1
                continue

            vec = normalize_skeleton(kpts)
            subject_data[subject_id].append(vec)

    print(f"  wczytano {total} wierszy, odrzucono {skipped_low_conf} (niska pewność), "f"{skipped_bad_hip} (złe biodra)")
    print(f"  osoby: {len(subject_data)}")
    return subject_data


def save_npz(subject_data, out_path):
    # npz nie lubi słowników wprost, więc spłaszczamy do subject_id + tablicy
    all_ids = []
    all_vecs = []
    for sid, vecs in subject_data.items():
        for v in vecs:
            all_ids.append(sid)
            all_vecs.append(v)

    np.savez_compressed(
        out_path,
        ids=np.array(all_ids),
        vecs=np.array(all_vecs, dtype=np.float32),
    )
    print(f"  zapisano {len(all_ids)} wektorów -> {out_path}")


if __name__ == "__main__":
    import os

    base = "gait_data"
    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(base, f"casia-b_pose_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"pomijam {split}, brak pliku")
            continue
        print(f"przetwarzam {split}...")
        data = process_csv(csv_path)
        save_npz(data, os.path.join(base, f"{split}.npz"))