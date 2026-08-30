# Informacje pliku database.py

## Cel programu

Program ma za zadanie zapisywać embeddingi i metadane wyciętych obiektów w plikach `.npy`, organizować je w folderach tematycznych oraz utrzymywać logiczny porządek i historię tożsamości. Docelowo program ma wspierać naukę modeli w trybie offline. Plik `identities.json` jest tworzony w celu utrzymania globalnego porządku ID w całej sesji, co ma pozwolić na odtwarzanie tożsamości po ponownym uruchomieniu programu.

## Sposób działania funkcji
    _init_(self, config_path: str = "config.yaml"):
        
