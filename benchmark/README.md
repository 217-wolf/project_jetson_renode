# Benchmark

## Setup (first time)

Requires Python 3.11+ (tested on 3.11 and 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

dependencies included in the project's root `requirements.txt`


### Model weights (not included in repo)
Place `.pt` files as referenced in `benchmark/configs/benchmark.yaml`:


## to run benchmark:
### if precision, recall, f1, mAP50 = 0 -> probable unavailability of labels (Ground Truth)
python -m benchmark.run_benchmark

## to generate plots and summary dashboard:
python -m benchmark.plot_results
Outputs to `benchmark/reports/plots/`

## to run manual VisDrone dataset analysis:
yolo detect val model=modele/yolo26n.pt data=VisDrone.yaml
yolo detect val model=modele/yolo8n.pt data=VisDrone.yaml


## dataset config /benchmark/configs/benchmark.yaml
### for smoke test
Without external dataset, only the `smoke` test config (4 sample images in
`benchmark/dataset/smoke/`) works — enough to verify the pipeline runs,
but won't produce meaningful precision/recall/mAP numbers.

dataset:
  name: smoke
  path: benchmark/dataset/smoke/images
  labels: benchmark/dataset/smoke/labels
  
### for VisDrone
The benchmark expects VisDrone dataset **outside** the repo folder:
`../datasets/VisDrone/images/val/`
`../datasets/VisDrone/labels/val/`

Download: https://github.com/VisDrone/VisDrone-Dataset

dataset:
  name: visdrone
  path: ../datasets/VisDrone/images/val
  labels: ../datasets/VisDrone/labels/val


## adding a new model to the benchmark

### if it's a YOLO model (Ultralytics-compatible .pt)
No code changes needed. Just add an entry to `benchmark/configs/benchmark.yaml`:

```yaml
experiments:
  - name: my_new_model_640
    model: path/to/my_model.pt
    imgsz: 640
```

Requirements for the model file:
- Must be loadable via `ultralytics.YOLO(path)`
- Must output `person`/`pedestrian`-equivalent class detections (check `model.names`)
- If class names differ from COCO, update `benchmark/metrics/class_mapping.py`

### if it's NOT a YOLO model (e.g. EfficientNet, OSNet, custom architecture)
Not supported yet. The current `YOLOBenchmarkRunner` is hardcoded to the
Ultralytics YOLO API (`.boxes`, `.xyxy`, `.conf`, `.cls`).