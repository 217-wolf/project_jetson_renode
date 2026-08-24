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