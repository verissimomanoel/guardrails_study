import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generate_metrics(result_name):
    file_name = f"results/input/results_{result_name}.csv"
    if os.path.exists(file_name):
        df_results = pd.read_csv(file_name)

        if "toxic_chat" not in result_name:
            y_true = np.zeros(len(df_results), dtype=int)
            false_negative = 0
        else:
            y_true = df_results["toxicity"].values
            false_negative = (len(df_results.query("result == 1 and toxicity == 0")) / len(df_results)) * 100

        accuracy = accuracy_score(y_true, df_results["result"].values) * 100
        precision = precision_score(y_true, df_results["result"].values, average="macro", zero_division=0) * 100
        recall = recall_score(y_true, df_results["result"].values, average="macro", zero_division=0) * 100
        f1 = f1_score(y_true, df_results["result"].values, average="macro", zero_division=0) * 100
        lattency = df_results["lattency"].mean()

        return accuracy, precision, recall, f1, false_negative, lattency
    else:
        return 0, 0, 0, 0, 0, 0


def convert_toxic_chat():
    from datasets import load_dataset

    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    df_data = ds["test"].to_pandas()

    df_data.to_csv("data/toxic_chat.csv", index=False)

def main(run = 1):
    evals = [
        {
            "framework": "llm_guard",
            "dataset": "do_not_answer_en"
        },
        {
            "framework": "llama_guard",
            "dataset": "do_not_answer_en"
        },
        {
            "framework": "moderation",
            "dataset": "do_not_answer_en"
        },
        {
            "framework": "llm_guard",
            "dataset": "toxic_chat"
        },
        {
            "framework": "llama_guard",
            "dataset": "toxic_chat"
        },
        {
            "framework": "moderation",
            "dataset": "toxic_chat"
        }
    ]

    results = []
    for item in evals:
        accuracy, precision, recall, f1, false_negative, lattency = generate_metrics(f"{item["framework"]}_{item["dataset"]}_run_{run}")
        if accuracy != 0 and precision != 0 and recall != 0 and f1 != 0 and lattency != 0:
            result = {
                "run": run,
                "framework": item["framework"],
                "dataset": item["dataset"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_negative": false_negative,
                "lattency": lattency
            }
            results.append(result)

    # print(f"##### RUN - {run}")
    # print(json.dumps(results, indent=4))
    return results


if __name__ == "__main__":
    statistics =[]
    for run in range(1, 11):
        statistics.extend(main(run))

    df_result = pd.DataFrame.from_dict(statistics)
    df_result.to_csv("results.csv", index=False)

    df_statistics = pd.DataFrame()
    frameworks = df_result["framework"].unique()
    datasets = df_result["dataset"].unique()
    statistics = []

    for framework in frameworks:
        for dataset in datasets:
            df_filter = df_result.query(f"dataset == '{dataset}' and framework == '{framework}'")
            result = {
                "framework": framework,
                "dataset": dataset,
                "accuracy_mean": round(df_filter["accuracy"].mean(), 4),
                "accuracy_std": round(df_filter["accuracy"].std(), 4),
                "precision_mean": round(df_filter["precision"].mean(), 4),
                "precision_std": round(df_filter["precision"].std(), 4),
                "recall_mean": round(df_filter["recall"].mean(), 4),
                "recall_std": round(df_filter["recall"].std(), 4),
                "f1_mean": round(df_filter["f1"].mean(), 4),
                "f1_std": round(df_filter["f1"].std(), 4),
                "false_negative_mean": round(df_filter["false_negative"].mean(), 4),
                "false_negative_std": round(df_filter["false_negative"].std(), 4),
                "lattency_mean": round(df_filter["lattency"].mean(), 4),
                "lattency_std": round(df_filter["lattency"].std(), 4)
            }

            statistics.append(result)

    df_statistics = pd.DataFrame.from_dict(statistics)
    df_statistics.to_csv("statistics.csv", index=False)
    print(df_statistics.head())
    # convert_toxic_chat()

