from pathlib import Path
import csv
import yaml

from runners.yolo_runner import YOLOBenchmarkRunner

REPORT = "benchmark/reports/results.csv"

def load_config():

    with open(
        "benchmark/configs/benchmark.yaml"
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()
    models = config["models"]
    dataset = config["dataset"]["path"]
    results = []
    settings = config["settings"]
    labels_dir = config["dataset"].get("labels")

    for model in models:

        model_name = model["name"]
        model_path = model["path"]
        print(f"\nTesting {model_name}")

        runner = YOLOBenchmarkRunner(
            model_path,
            confidence=settings["confidence"],
            image_size=settings["image_size"],
            labels_dir=labels_dir
        )

        result = runner.run(dataset)
        result["name"] = model_name
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