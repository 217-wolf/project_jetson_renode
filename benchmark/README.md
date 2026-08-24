# Benchmark

## to run benchmark:
### if precision, recall, f1, mAP50 = 0 -> probable unavailability of labels (Ground Truth)
python -m benchmark.run_benchmark

## to run manual VisDrone dataset analysis:
yolo detect val model=modele/yolo26n.pt data=VisDrone.yaml
yolo detect val model=modele/yolo8n.pt data=VisDrone.yaml


## dataset config /benchmark/configs/benchmark.yaml
### for smoke test
dataset:
  name: smoke
  path: benchmark/dataset/smoke/images
  labels: benchmark/dataset/smoke/labels
  
### for VisDrone
dataset:
  name: visdrone
  path: ../datasets/VisDrone/images/val
  labels: ../datasets/VisDrone/labels/val



## Wnioski
Póki co : Modele COCO-pretrained (YOLOv8n, YOLO26n) mają ograniczoną skuteczność na aerial imagery z powodu bardzo małych obiektów (67% osób w VisDrone < 0.1% powierzchni obrazu). Przy IoU=0.5, standardowym progu dla COCO, recall spada do 0.15-0.19. Obniżenie IoU do 0.3 i confidence do 0.1 poprawia recall do ~0.33-0.38 kosztem precision. Sugeruje to, że modele wymagają fine-tuningu na danych aerial/SAR (np. VisDrone, SARD) zamiast bazowych wag COCO, jeśli mają być używane w dronach do wykrywania ludzi.