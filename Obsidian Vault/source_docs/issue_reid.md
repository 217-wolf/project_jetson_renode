
## Ważna diagnoza `UNKNOWN` dla osób

Aktualny problem znajduje się przede wszystkim w obsłudze plików bazy wzorców:

- `database.py` podczas ładowania szuka pliku `name.npy`.
- `database.py` podczas zapisu tworzy plik `name_class.npy`.
- Obecne pliki mają format `person_002_person.npy`, `bus_001_bus.npy` itd.
- W rezultacie metadane są odczytywane, ale embeddingi nie są znajdowane i lista wzorców pozostaje pusta.
- `matcher.py` dostaje pustą listę dla klasy `person` i zwraca `(None, 0.0)`.
- `gui.py` oraz `main.py` zamieniają brak nazwy na `UNKNOWN`.

Test w środowisku projektu potwierdził ten stan: baza zawiera klasy `bus`, `person` i `horse`, ale ładuje `0` wzorców.

## Dodatkowe miejsca do sprawdzenia przy naprawie ReID

- Ujednolicić jeden format nazwy pliku w całym `database.py` i sprawdzić istniejące pliki po zmianie.
- Upewnić się, że embedding wzorca i embedding z kamery pochodzą z tego samego ekstraktora. Obecne pliki osób mają rozmiar `(512,)`, a pliki innych obiektów `(1280,)`, co wskazuje na użycie różnych modeli.
- Sprawdzić preprocessing OSNet, w szczególności normalizację obrazu, ponieważ obecny przepływ dla OSNet różni się od typowego preprocessing'u modelu ReID.
- Rozdzielić dwa pojęcia: `instance_id` służy do śledzenia obiektu w czasie, a nazwa wzorca (`person_002` itd.) służy do rozpoznania konkretnej osoby.
- W trybach live w `gui.py` i `main.py` obecnie wykonywane jest dopasowanie nazwy, ale nie ma trackera nadającego stabilne ID między klatkami. Do pełnego ReID należy połączyć detekcję, embedding, dopasowanie do bazy i tracker.

Plik ten opisuje stan diagnostyczny. Nie zmienia działania programu.
