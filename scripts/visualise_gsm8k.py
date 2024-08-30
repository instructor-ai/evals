import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from glob import glob


def extract_scores(json_path: str):
    with open(json_path, "r") as file:
        data = json.load(file)
        for item in data:
            yield int(item["scores"]["ExactMatch"])


def bootstrap_batch(data, num_samples, sample_size):
    data = list(data)
    return [
        np.mean(np.random.choice(data, size=sample_size, replace=True))
        for _ in range(num_samples)
    ]


def generate_kde_plot(results: list[dict], visualisation_path: str, sample_size: int):
    accuracies = [item["scores"] for item in results]
    kdes = [stats.gaussian_kde(accuracy) for accuracy in accuracies]

    x_range = np.linspace(
        min([min(accuracy) for accuracy in accuracies]),
        max([max(accuracy) for accuracy in accuracies]),
        100,
    ).tolist()
    kdes = [kde(x_range) for kde in kdes]

    # Plot the KDEs
    plt.figure(figsize=(10, 6))
    for i, (kde, item) in enumerate(zip(kdes, results)):
        file_name = os.path.basename(item["file_name"])
        plt.plot(
            x_range,
            kde,
            label="Correct Answer" if "correct-answer.json" in file_name else "Answer",
        )
    plt.title(f"Kernel Density Estimation for bootstrap sample size {sample_size}")
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(visualisation_path)


def generate_boxplot(results: list[dict], visualisation_path: str):
    df = pd.DataFrame(
        [
            {"File": item["file_name"], "Accuracy": score}
            for item in results
            for score in item["scores"]
        ]
    )
    # Create a mapping for the labels
    df["File"] = df["File"].apply(
        lambda x: "Correct Answer" if "correct-answer.json" in x else "Answer"
    )

    # Sort the dataframe to ensure consistent ordering
    df = df.sort_values("File")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create the box plot
    sns.boxplot(x="File", y="Accuracy", data=df)

    # Customize the plot
    plt.title("Accuracy Distribution by File", fontsize=16)
    plt.xlabel("File", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45)

    # Add individual data points
    sns.stripplot(x="File", y="Accuracy", data=df, color="black", size=4, alpha=0.5)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(visualisation_path)


def compute_statistics(results: list[dict]):
    for item in results:
        mean = np.mean(item["scores"])
        std = np.std(item["scores"])
        var = np.var(item["scores"])
        yield {
            "file_name": item["file_name"],
            "mean": mean,
            "std": std,
            "var": var,
        }


if __name__ == "__main__":
    BOOTSTRAP_SAMPLES = 1000
    BOOTSTRAP_SAMPLE_SIZE = 200
    RESULTS_FILE = "./bootstrap_results.jsonl"
    DATA_DIR = "./scripts/data/raw"
    np.random.seed(42)

    # Read in the .json files and transform them into an input-output pair
    results = [
        {
            "file_name": f,
            "scores": bootstrap_batch(
                extract_scores(f), BOOTSTRAP_SAMPLES, BOOTSTRAP_SAMPLE_SIZE
            ),
        }
        for f in glob(f"{DATA_DIR}/*.json")
    ]

    generate_boxplot(results, "./boxplot.png")
    generate_kde_plot(results, "./kde.png", BOOTSTRAP_SAMPLE_SIZE)
    print(pd.DataFrame(compute_statistics(results)))
