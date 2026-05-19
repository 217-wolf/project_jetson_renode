# Projekt Rozpoznawania obiektów - Nvidia Jetson i Renode

## Cel projektu  

---

Projekt ma za zadanie doprowadzenie do rozpoznawania unikalnych obiektów z obrazu kamer na różnych urządzeniach NVIDIA Jetson, wykorzystując ustalony model widzenia komputerowego oraz technologię Antmicro Renode do rozpoznawania jednego obiektu mimo różnych osadzeń obrazu obiektu.

  

## Bootowanie trzech Jetson Orin Nano
---

Jetson1 wersja Jetson Orin Developer Kit 6.2.1

Jetson2 - wersja Jetson Orin Nano Developer Kit 5.1.3

Jetson3 - wersja Jetson Orin Developer Kit 5.1.3


## Postępy po bootwaniu

Jetson3 - instalacja VScode

Jetson1 - instalacja VScode, przeglądarka

  
## Kwestie podziału etapu II

dział - Model rozpoznawania obiektu

dział - Operacje Elektroniką

  
### Uczenie modelu

Wykorzystanie Modelu YOLO. Ustalenie typów obrazu do uczenia modeli. Sprawdzenie różnych modeli oraz jak się sprawdzają. Ewentualne douczanie

  

testowanie i porównanie

  

Implementacja najlepiej rozpoznającego modelu (rozponanie osoby z różnych perspektyw)

  

### Elektronika

Proste podpięcia - np. kamery do jetson-ów

"Złaczenie" modeli do jetsonów i ich konfiguracja.

próbny film do rozpoznawania obrazów

  

wykorzystanie renode'a do synchronizacji przetworzeń kilku jetsonów

  

Rozwiązywanie potencjalnych problemów końcowych

  

### Walidacja działania systemu jetsonów względem założeń początkowych

  

## Przy pomyślnej walidacji zakończenie projektu