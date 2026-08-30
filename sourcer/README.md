# Sourcer

Folder `sourcer` zawiera główne moduły systemu detekcji, ekstrakcji cech, rozpoznawania i obsługi kamery.

## Ogólne zadanie

System pobiera obraz z kamery lub pliku, wykrywa obiekty za pomocą YOLO, wycina ich fragmenty, tworzy embeddingi i porównuje je ze wzorcami zapisanymi w bazie. Wynik może być pokazany w oknie, zapisany do pliku albo wykorzystany do śledzenia instancji obiektu.

## Moduły

- `__init__.py` - oznacza folder jako pakiet Pythona.
- `analyzer.py` - analizuje obraz, tworzy embeddingi i nadaje ID obiektom znajdującym się na jednym obrazie.
- `camera.py` - otwiera kamerę, odczytuje klatki i zarządza jej zamykaniem.
- `camera_gui.py` - udostępnia prosty interfejs do uruchamiania i zatrzymywania automatycznego zbierania embeddingów oraz podglądu logu i podsumowania.
- `collect_embeddings.py` - uruchamia ciągłą detekcję z kamery, śledzi obiekty, nadaje ID i zapisuje embeddingi.
- `database.py` - ładuje, zapisuje, dodaje i usuwa wzorce embeddingów.
- `detector.py` - wykonuje detekcję obiektów modelem YOLO.
- `embedding_logger.py` - zapisuje informacje o zapisanych embeddingach do pliku CSV i odczytuje podsumowania.
- `extractor.py` - tworzy embeddingi z wyciętych fragmentów obrazu; dla osób może używać OSNet.
- `gui.py` - główny interfejs aplikacji do dodawania wzorców, testów, listy wzorców i szybkiej analizy.
- `main.py` - główny punkt wejścia i wybór trybu pracy programu.
- `matcher.py` - porównuje embedding obiektu z embeddingami wzorców i wybiera najlepsze dopasowanie.
- `visualizer.py` - rysuje ramki, klasy, ID, pewność detekcji i wynik rozpoznania na obrazie.

## Główny przepływ rozpoznawania

1. `CameraManager` pobiera klatkę obrazu.
2. `ObjectDetector` wykrywa obiekty i zwraca ich klasy, ramki oraz pewność YOLO.
3. `FeatureExtractor` tworzy embedding dla każdego wyciętego obiektu.
4. `PatternsDatabase` udostępnia wzorce należące do wykrytej klasy.
5. `ObjectMatcher` oblicza podobieństwo i wybiera nazwę wzorca albo zwraca brak dopasowania.
6. `Visualizer` przedstawia wynik na obrazie.

Pewność YOLO określa, czy fragment wygląda jak dana klasa, na przykład `person`. Nie jest ona identyfikacją konkretnej osoby. Do identyfikacji potrzebne jest osobne, poprawne dopasowanie embeddingu przez `ObjectMatcher`.

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
