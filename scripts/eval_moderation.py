import os
import pickle
import time
from datetime import datetime

import pandas as pd
### OpenAI
from openai import OpenAI
from tqdm.auto import tqdm

client = OpenAI()

# dataset_name = "do_not_answer_en"
# column = "question"

dataset_name = "toxic_chat"
column = "user_input"

result_name = f"results_moderation_{dataset_name}"

df_data = pd.read_csv(f"../data/{dataset_name}.csv")

pickle_name = "moderation_control.pickle"

if os.path.exists(pickle_name):
    with open(pickle_name, 'rb') as handle:
        moderation_control = pickle.load(handle)
else:
    moderation_control = {}
    with open(pickle_name, 'wb') as handle:
        pickle.dump(moderation_control, handle, protocol=pickle.HIGHEST_PROTOCOL)

def apply_safeguards(input_prompt = ""):
    now = datetime.now()
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=input_prompt,
    )
    end = datetime.now()
    lattency = (end - now).total_seconds()

    safe = 1
    unsafe = 0

    if dataset_name == "toxic_chat":
        safe = 0 # Toxicity = 0
        unsafe = 1 # Toxicity = 1

    results = response.results[0]
    if results.flagged:
        categories = results.categories.model_dump()
        guardrails = [key for key in categories.keys() if categories[key]]
        return unsafe, ",".join(str(x) for x in guardrails), lattency
    else:
        return safe, "", lattency


def run_moderation(folder = "input", run = 1):
    results = []
    guardrails = []
    lattencies = []
    for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        key_moderation = f"{result_name}_run_{run}_{index}"
        if key_moderation not in moderation_control.keys():
            safe, guardrail, lattency = apply_safeguards(row[column])

            moderation_control[key_moderation] = {
                "safe": safe,
                "guardrail": guardrail,
                "lattency": lattency
            }

            with open(pickle_name, 'wb') as handle:
                pickle.dump(moderation_control, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if index % 100 != 0:
                time.sleep(1)
            else:
                time.sleep(40)
        else:
            safe = moderation_control[key_moderation]["safe"]
            guardrail = moderation_control[key_moderation]["guardrail"]
            lattency = moderation_control[key_moderation]["lattency"]

        results.append(safe)
        guardrails.append(guardrail)
        lattencies.append(lattency)

    df_data["result"] = results
    df_data["guardrails"] = guardrails
    df_data["lattency"] = lattencies
    df_data.to_csv(f"results/{folder}/{result_name}_run_{run}.csv", index=False)


if __name__ == "__main__":
    for epoch in range(1, 11):
        run_moderation("input", epoch)