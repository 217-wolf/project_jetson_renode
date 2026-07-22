import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.dataset import TripletGaitDataset
from src.model import EmbeddingNet, TripletLoss

def train(epochs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    #całe datassety opakowane w potrzebne metody
    train_datasest = TripletGaitDataset('gait_data/train.npz', 50000)
    valid_dataset = TripletGaitDataset('gait_data/valid.npz', 5000)

    #1024 różne triplety
    train_loader = DataLoader(train_datasest, batch_size=1024, shuffle=True, num_workers=2, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, num_workers=2, drop_last=True)

    model = EmbeddingNet().to(device)
    criterion = TripletLoss(margin=0.3) #to ocenia czy git czy nie git
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #lr jest domyślne
    best_valid_loss = float("inf")

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        active_loss = 0.0

        #liczenie gradientów i optymalizacja wag
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_anch = model(anchor)
            emb_pos = model(positive)
            emb_neg = model(negative)

            loss = criterion(emb_anch, emb_pos, emb_neg)
            train_loss += loss.item()

            with torch.no_grad():
                dist_pos = F.pairwise_distance(emb_anch, emb_pos)
                dist_neg = F.pairwise_distance(emb_anch, emb_neg)
                #Jezeli odleglosc miedzy positive i negative + margines jest wiekszy niz 0 to znaczy ze siec nie odroznia jeszcze tego tripletu:
                #   małe active_loss -> siec dobrze odroznia wiekszosc par
                #   duze active loss -> siec myli sie w wielu przypadkach
                active = (dist_pos - dist_neg + criterion.margin > 0).float().mean() #float zmienia True na 1.0 i False na 0.0
                active_loss += active.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #walidacja
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in valid_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                emb_anch = model(anchor)
                emb_pos = model(positive)
                emb_neg = model(negative)

                loss = criterion(emb_anch, emb_pos, emb_neg)
                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        active_loss /= len(train_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), r"embedding_net_best.pt")

        print(f"epoka {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "f"valid_loss={valid_loss:.4f}  aktywne_triplety={active_loss:.2%}")

if __name__ == '__main__':
    train(epochs=20)