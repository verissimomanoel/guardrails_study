from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

from huggingface_hub import login
# login().

dataset_name = "do_not_answer_en"
column = "question"

# dataset_name = "toxic_chat"
# column = "user_input"

result_name = f"results_llama_guard_{dataset_name}"

# model_id = "meta-llama/Llama-Guard-3-8B"
model_id = "meta-llama/Llama-Guard-3-8B-INT8"
device = "cuda"
dtype = torch.bfloat16

## quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
    llm_int8_enable_fp32_cpu_offload = True,
)

safeguards_map = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse"
}

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)#, quantization_config = bnb_config)

def moderate_with_template(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

df_data = pd.read_csv(f"../data/{dataset_name}.csv")

def apply_safeguards(input_prompt = ""):
    chat = [
        {"role": "user", "content": input_prompt}
    ]

    now = datetime.now()
    result = moderate_with_template(chat).replace("\n", "")
    end = datetime.now()
    lattency = (end - now).total_seconds()

    safe = 1
    unsafe = 0

    if dataset_name == "toxic_chat":
        safe = 0 # Toxicity = 0
        unsafe = 1 # Toxicity = 1

    if result == "safe":
        return safe, "", lattency
    else:
        code = result.replace("unsafe", "").upper()
        return unsafe, safeguards_map[code], lattency


def run_safeguards(folder = "input", run = 1):
    results = []
    guardrails = []
    lattencies = []
    for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        safe, guardrail, lattency = apply_safeguards(row[column])
        results.append(safe)
        guardrails.append(guardrail)
        lattencies.append(lattency)

    df_data["result"] = results
    df_data["guardrail"] = guardrails
    df_data["lattency"] = lattencies
    df_data.to_csv(f"results/{folder}/{result_name}_run_{run}.csv", index=False)


if __name__ == "__main__":
    # https://platform.openai.com/docs/guides/moderation/quickstart
    for run in range(1, 11):
        print(f"##### RUN - {run}")
        run_safeguards("input", run)