import numpy as np
from collections import defaultdict
import torch

from torch.utils.data import Dataset
import random

class TripletGaitDataset(Dataset):
    """
    Losowy sampling tripletów: anchor i positive to dwa różne szkielety tej samej osoby
    negative to szkielet innej losowej osoby.

    length -> ile tripletów "udajemy" że ma epoka; ponieważ sampling jest losowy, to nie jest prawdziwa liczba unikalnych par, tylko sterowanie
    długością epoki.
    """

    def __init__(self, npz_path, length):
        data = np.load(npz_path)
        ids = data['ids']
        vecs = data['vecs']

        freq_dict = defaultdict(list)

        for i in range(len(ids)):
            freq_dict[ids[i]].append(vecs[i])

        to_remove = set()
        for key, item in freq_dict.items():
            if len(item) < 2:
                to_remove.add(key)

        for i in to_remove:
            del freq_dict[i]

        self.data = freq_dict
        self.ids = list(freq_dict.keys())
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anchor_id, negative_id = random.sample(self.ids, 2) # anchor + positive to ta sama osoba, ale inne klatki

        anch_val, pos_val = random.sample(self.data[anchor_id], 2)
        neg_val = random.choice(self.data[negative_id])

        return (torch.from_numpy(anch_val).float(),
                torch.from_numpy(pos_val).float(),
                torch.from_numpy(neg_val).float())