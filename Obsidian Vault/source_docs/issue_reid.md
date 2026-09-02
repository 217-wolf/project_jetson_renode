
## Zapis osadzeń z pamięci RAM do `patterns_database`

Należy zaprojektować i wdrożyć sposób trwałego zapisu osadzeń (embeddingów) przechowywanych tymczasowo w pamięci RAM do folderu `patterns_database`. Zapis powinien obejmować przede wszystkim embeddingi osób. Dla klasy reprezentującej osoby należy utworzyć osobny folder w `patterns_database`, co pozwoli zachować uporządkowaną strukturę projektu i ułatwi ewentualną zmianę sposobu przechowywania danych w przyszłości.

Do rozstrzygnięcia pozostaje, jaki wariant danych powinien być utrwalany dla jednej osoby/klasy:

- pojedyncze osadzenie reprezentatywne (najlepsze)
- prototyp utworzony z wielu obserwacji (uśrednione osadzenie)
- galeria osadzeń obejmująca różne kąty i warunki obserwacji
- kombinacja 3 opcji powyżej

Jeżeli dla jednej osoby/klasy będzie zapisywany więcej niż jeden plik osadzenia, pliki nie powinny znajdować się bezpośrednio w głównym folderze `patterns_database/`. Dla każdej instancji należy utworzyć osobny podfolder zawierający wyłącznie osadzenia tej instancji oraz powiązane metadane. Przykładowo: folder `person/` dla klasy a w nim podfoldery `person_1/` , `person_2/` itd.

W wariancie z wieloma plikami osadzeń wymagany jest plik JSON z metadanymi, zawierający:

- nazwę klasy.
- identyfikator instancji
- wskazanie najlepszego osadzenia / prototyp
- poziom pewności (confidence level)

\
\
Przykładowa organizacja:
```
patterns_database/
└── <instance_id>/
    ├── best_instance_id_embedding.npy
    ├── instance_id_embedding_001.npy
    ├── instance_id_embedding_002.npy
    ├── prototype_instance_id.npy
    └── instance_id_metadata.json
```