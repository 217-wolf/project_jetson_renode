import numpy as np

punkty_na_ciele = {
    0: "nos",
    1: "lewe_oko",
    2: "prawe_oko",
    3: "lewe_ucho",
    4: "prawe_ucho",
    5: "lewy_bark",
    6: "prawy_bark",
    7: "lewy_lokiec",
    8: "prawy_lokiec",
    9: "lewy_nadgarstek",
    10: "prawy_nadgarstek",
    11: "lewe_biodro",
    12: "prawe_biodro",
    13: "lewe_kolano",
    14: "prawe_kolano",
    15: "lewa_kostka",
    16: "prawa_kostka"
}

def normalize_skeleton(kpts):
    """
    Czyli jak daleko punkty są oddalone od centrum bioder
    """
    mid = (kpts[11] + kpts[12]) / 2
    kpts_centered = kpts - mid

    scale = np.linalg.norm(kpts_centered[0]) +1e-6
    kpts_normalized = kpts_centered / scale

    return kpts_normalized.flatten()