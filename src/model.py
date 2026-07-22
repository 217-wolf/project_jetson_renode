import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """
    Liczby 128 / 64 wpisałem arbitralnie, te hiperparametry trzeba jeszcze dostosować.
    Używamy normy L2 z dwóch powodów:
        - Jeżeli embedingi leżą w sferze jednostkowej to cos similarity to zwykly iloczyn
        - Spójność z TripletLoss (domyślnie liczy odleglość euklidesową)
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(34, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
        )

    def forward(self, X):
        embedding = self.layers(X)
        return F.normalize(embedding, p = 2, dim = 1) #dim = 1 oznacza normalizacje dla kazdego wiersza osobno

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Odległości euklidesowe o której mowa w EmbeddingNet
        dist_pos = F.pairwise_distance(anchor, positive)  # ta sama osoba
        dist_neg = F.pairwise_distance(anchor, negative)  # inna osoba

        # max(0, dist_pos - dist_neg + margin)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean() #dla stablinego kroku optymalizacji