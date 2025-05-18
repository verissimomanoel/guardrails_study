# LLM Guard Evaluation Experiments
[![Linguagem](https://img.shields.io/badge/linguagem-Python-blue.svg)](https://www.python.org/)
[![Licença](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains scripts for evaluating different approaches to content moderation and safety guards in Language Models (LLMs).

## Overview

The project evaluates three different guard systems:
- LLaMA Guard models
- LLM Guard library
- OpenAI Moderation API

Each system is tested on two datasets:
- `do_not_answer_en`: A collection of prompts that should be rejected by responsible AI systems
- `toxic_chat`: A dataset of texts to identify as toxic or safe

## Scripts

The evaluation process consists of 4 main scripts that should be executed in sequence:

1. `eval_llama_guards.py`: Evaluates Meta's Llama-Guard-3-8B model
2. `eval_llm_guards.py`: Tests the LLM Guard library with BanTopics and Toxicity scanners
3. `eval_moderation.py`: Evaluates OpenAI's Moderation API
4. `generate_results.py`: Generates final results, metrics, and statistical analysis

## Dataset Configuration

For each script (except generate_results.py), you need to set the appropriate dataset configuration variables at the beginning of the file:

### For do_not_answer_en dataset:
dataset_name = "do_not_answer_en"

column = "question"

### For toxic_chat dataset:
dataset_name = "toxic_chat"

column = "user_input"

The scripts have these variables defined at the top, where you can uncomment/comment the appropriate configuration.

## Requirements

- Python 3.12.0
- Required packages:
    All the packages are in a requirments.txt file.

## Running the Experiments

1. Modify the dataset configuration in each script by setting the appropriate variables
2. Run each script in sequence:

```bash
cd scripts
python eval_llama_guards.py
python eval_llm_guards.py
python eval_moderation.py
```

3. After run all the scripts above for the two datasets, run the script for the consolidate the results:
```bash
python generate_results.py
```

## Results
The scripts will generate:
- Individual run results in the directory `results/input/`
- Aggregated results in `results.csv`
- Statistical summary in `statistics.csv`

The metrics include accuracy, precision, recall, F1 score, false negative rate, and latency for each guard system and dataset.

This README provides clear instructions about:
1. The purpose of the project
2. The scripts and their execution order
3. How to configure the dataset variables for both datasets
4. The requirements and expected outputs