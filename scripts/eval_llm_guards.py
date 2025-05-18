import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

### LLM Guard inputs
from llm_guard import scan_prompt
from llm_guard.input_scanners import BanTopics
from llm_guard.input_scanners import Toxicity
from llm_guard.input_scanners.toxicity import MatchType

dataset_name = "do_not_answer_en"
column = "question"

# dataset_name = "toxic_chat"
# column = "user_input"

result_name = f"results_llm_guard_{dataset_name}"

df_data = pd.read_csv(f"../data/{dataset_name}.csv")

class CFG:
    topics_list = ["sexual", "violence"]

    ### input scanners
    inp_topics_thres = 0.85
    inp_toxic_thres = 0.85


### define Ban Topics safeguard
inp_scan_ban_topics = BanTopics(topics=CFG.topics_list, threshold=CFG.inp_topics_thres)

### define Toxicity safeguard
inp_scan_toxic = Toxicity(threshold=CFG.inp_toxic_thres, match_type=MatchType.FULL)

### define input scanners pipeline
input_scanners = [
    inp_scan_ban_topics,
    inp_scan_toxic,
]

def apply_safeguards(input_prompt = "", inp_scanners = input_scanners):
    now = datetime.now()
    sanitized_prompt_input, results_valid_input, results_score_input = scan_prompt(
        inp_scanners, input_prompt, fail_fast=True
    )
    end = datetime.now()
    lattency = (end - now).total_seconds()

    safe = 1
    unsafe = 0

    if dataset_name == "toxic_chat":
        safe = 0 # Toxicity = 0
        unsafe = 1 # Toxicity = 1

    if any(not result for result in results_valid_input.values()):
        keys = [*results_valid_input.keys()]
        return unsafe, keys[0], lattency
    else:
        return safe, "", lattency


def run_llm_guard(folder = "input", run = 1):
    results = []
    guardrails = []
    lattencies = []
    for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        safe, guardrail, lattency = apply_safeguards(row[column], input_scanners)
        results.append(safe)
        guardrails.append(guardrail)
        lattencies.append(lattency)
    df_data["result"] = results
    df_data["guardrails"] = guardrails
    df_data["lattency"] = lattencies
    df_data.to_csv(f"results/{folder}/{result_name}_run_{run}.csv", index=False)


if __name__ == "__main__":
    # {'BanTopics': 1.0}
    for epoch in range(1, 11):
        run_llm_guard("input", epoch)