from pathlib import Path
import csv
import yaml

from benchmark.runners.yolo_runner import YOLOBenchmarkRunner

REPORT = "benchmark/reports/results.csv"

def load_config():

    with open("benchmark/configs/benchmark.yaml") as file:
        return yaml.safe_load(file)


def main():

    config = load_config()
    experiments = config["experiments"]
    dataset = config["dataset"]["path"]
    results = []
    settings = config["settings"]
    labels_dir = config["dataset"].get("labels")

    for experiment in experiments:

        experiment_name = experiment["name"]
        model_path = experiment["model"]
        image_size = experiment["imgsz"]
        print(f"\nTesting {experiment_name}")

        runner = YOLOBenchmarkRunner(
            model_path,
            confidence=settings["confidence"],
            image_size=image_size,
            labels_dir=labels_dir,
            iou_threshold=settings["iou_threshold"]
        )

        result = runner.run(dataset)
        result["experiment"] = experiment_name
        result["model_path"] = model_path
        result["dataset"] = config["dataset"]["name"]
        result["imgsz"] = image_size
        result["confidence_threshold"] = settings["confidence"]
        result["iou_threshold"] = settings["iou_threshold"]
        results.append(result)

    Path("benchmark/reports").mkdir(
        exist_ok=True
    )

    with open(REPORT, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", REPORT)

if __name__ == "__main__":
    main()