# Skeleton Re-ID


Szczegółowy opis projektu znajduje się w [docs/raport.md](docs/raport.md).

## Instalacja

Projekt wymaga Pythona 3.13.

```powershell
git clone REPO
cd project_jetson_renode

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Przygotowanie danych

Pobierz zbiór iLIDS-VID i umieść go w katalogu:

```text
data/raw/ilids_vid/i-LIDS-VID/sequences/
├── cam1/
└── cam2/
```

Następnie uruchom:

```powershell
python .\scripts\prepare_ilids_dataset.py
```

Przygotowane dane zostaną zapisane w katalogu:

```text
data/processed/ilids_pose/
```

## Sprawdzenie danych

```powershell
python .\scripts\audit_ilids_dataset.py
```

## Trening modelu

```powershell
python .\scripts\train_reid.py
```

Najlepszy model zostanie zapisany jako:

```text
models/skeleton_reid_best.pt
```

Ostatni checkpoint zostanie zapisany jako:

```text
models/skeleton_reid_last.pt
```

## Wznowienie treningu

```powershell
python .\scripts\train_reid.py --resume .\models\skeleton_reid_last.pt
```

## Testy

```powershell
python -m unittest discover -s tests -v
```

## Wynik

Najlepszy uzyskany wynik walidacyjny:

```text
Rank-1: 15%
```

