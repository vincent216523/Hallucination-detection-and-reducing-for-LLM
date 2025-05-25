import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

# Global variable
debug = True

# Prompt templates
prompts = {
    'default': Template(
        "Q: $question\nA: "
    ),
    'sys-wrong': Template(
        "<<SYS>> Your answer was wrong. Retry answering the question.<</SYS>>\nQ: $question\nA: "
    ),
    'inst-wrong': Template(
        "<<INST>> Your answer was wrong. Retry answering the question.<</INST>>\nQ: $question\nA: "
    ),
    'rephrase': Template(
        "<<SYS>> Rephrase the question.<</SYS>>\n $question"
    ),
    'hint': Template(
        "<<SYS>> Provide a hint to the question.<</SYS>>\n\n Q: $question\n\n Hint: "
    ),
    'hint-qa': Template(
        "Hint: $hint\n Q: $question\n A: "
    ),
}

prompt_splitter = {
    'default': "A: ",
    'sys-wrong': "A:",
    'inst-wrong': "<<SYS>>",
    'rephrase': "<<SYSS>>",
    'hint': "### Hint: ",
}

def load_prompts():
    return prompts

#Load Datasets
def load_data(start=0, end=-1):
    cwd = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(cwd), 'data')
    trivia_dataset = load_dataset('trivia_qa', data_dir='rc.nocontext', cache_dir=data_dir)
    full_dataset = []
    for obs in tqdm(trivia_dataset['train']):
        aliases = []
        aliases.extend(obs['answer']['aliases'])
        aliases.extend(obs['answer']['normalized_aliases'])
        aliases.append(obs['answer']['value'])
        aliases.append(obs['answer']['normalized_value'])
        full_dataset.append((obs['question'], aliases))
    dataset = full_dataset[start: end]

    del trivia_dataset
    return dataset

# format prompt
def format_prompt(original_prompt: str, question:str,  answer:str, template_name:str, num_answer_str = 20):
    template = prompts[template_name]
    if num_answer_str > 0:
        gen_len = min(len(answer), num_answer_str)
        answer = answer[:gen_len]
    prompt = original_prompt + answer + '\n' + template.substitute(
        question=question
    )
    return prompt

# rephrase propmt with llm model
def rephrase_prompt(model, tokenizer, question:str, template_name:str,max_length = 20, sensitivity = 0.01):
    rephrase_prompt = prompts[template_name].substitute(
        question=question
    )
    input_ids = tokenizer.encode(rephrase_prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_length, num_return_sequences=1, do_sample=True, temperature=sensitivity)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = output[len(rephrase_prompt):].strip()
    if template_name == 'hint':
        if "### Question" in response:
            response = response.split("### Question")[0].strip()
        elif "Q:" in response:
            response = response.split("Q:")[0].strip()
        elif "Question:" in response:
            response = response.split("Question:")[0].strip()
        elif "\n" in response and len(response.split("\n")) > 1:
            response = response.split("\n")[0].strip()
        elif "Hint:" in response and len(response.split("Hint:")) > 1:
            response = response.split("Hint:")[0].strip()
        output = prompts['hint-qa'].substitute(
            question=question,
            hint=response
        )
    elif template_name == 'rephrase':
        output = prompts['default'].substitute(
            question=response
        )

    return output

# format results generated from generate attributes
def format_result(result,save_all = False):

    if save_all:
        result['question'] = result['question'][0]
        result['str_response'] = result['str_response'][0]
        return result
    else:
        save_result = {}
        save_result['question'] = result['question'][0]
        save_result['str_response'] = result['str_response'][0]
        save_result['hallucination'] = result['hallucination']
        save_result['correct'] = result['correct']
        save_result['hallucination_prob'] = result['hallucination_prob']
        save_result['turn'] = result['turn']
        return save_result

def open_json(save_path):
    results = []
    with open(save_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())  # Deserialize each line
            results.append(result)
    return results

def read_pickle(save_path):
    results = []
    with open(save_path, 'rb') as f:
        while True:
            try:
                result = pickle.load(f)
                results.append(result)
            except EOFError:
                break
    return results
