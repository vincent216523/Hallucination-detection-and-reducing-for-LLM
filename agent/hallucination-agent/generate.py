import argparse
import gc
import os
import pickle
from datetime import datetime

import numpy as np
import torch

from datasets import load_dataset
from tqdm import tqdm

from utils.classifier import run_classifier, aggregate_pred
from utils.data_utils import (format_prompt, format_result, load_data,
                              load_prompts, rephrase_prompt)
from utils.model_utils import generate_attributes, load_model

from models.load_model import load
from models.classifer_softmax import SoftMaxClassifier
from models.ig import RNNHallucinationClassifier

#Flags
debug = True
data_mining = True

# global variables
cwd = os.path.dirname(__file__)
results_dir = os.path.join(cwd, 'result')
# create directory if not exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

template = load_prompts()

# generate N-turns
def generate_multiturn_attributes(model, tokenizer, embedder, start_template, question, aliases, n,p_thresh,save_path,idx,sensitivity):
    if debug: print("Generating N-turns...")
    n_gen_result = []
    response = []
    prompt = start_template.substitute(question=question)
    for i in range(n):
        # generate attributes
        with torch.no_grad():
            results = generate_attributes(model, tokenizer, embedder, prompt, aliases)
        results['turn'] = i
        results['idx'] = idx

        # call classifier inference
        template_name = 'sys-wrong'
        classifiers = ['RNN']
        classifier_model = [load(i) for i in classifiers]
        pred = []
        for j in range(len(classifier_model)):
            try:
                p_classifier = run_classifier(classifier_model[j], results, classifiers[j], val_fix = -1, rand = False)
                pred.append(p_classifier)
                print(f"Classifier {classifiers[j]} - Hallucination: {p_classifier}")
            except Exception as e:
                print(f"Error in classifier {classifiers[j]}: {classifier_model[j]}")
                print(e)
        prob = aggregate_pred(pred, agg='max')
        results['hallucination_prob'] = prob

        if prob > p_thresh:
            results['hallucination'] = True
            prompt = rephrase_prompt(model, tokenizer, question, 'hint' , max_length = 20,sensitivity = sensitivity)
            #prompt = format_prompt(results['question'][0], question,  results['str_response'][0], template_name , num_answer_str = 10)
            if data_mining:
                n_gen_result.append(format_result(results, save_all=True))
            else:
                n_gen_result.append(format_result(results, save_all=False))
        else:
            results['hallucination'] = False
            if data_mining:
                n_gen_result.append(format_result(results, save_all=True))
            else:
                n_gen_result.append(format_result(results, save_all=False))
            break


        if debug:
            print(f"Turn {i}:")
            print(f"Question: {results['question']}")
            print(f"Response: {results['str_response']}")
            print(f"Hallucination: {results['hallucination']}")
            print(f"Correct: {results['correct']}")
            print(f"Next Turn Prompt: {prompt}")
            print(f"Hallucination Prob: {prob}")
        del results
        gc.collect()

    # Save the result to disk
    save_result([n_gen_result], save_path)
    
    if debug:
        for i in range(len(n_gen_result)):
            print(f"Final Turn {i}:")
            print(f"Question: {n_gen_result[i]['question']}")
            print(f"Response: {n_gen_result[i]['str_response']}")
            print(f"Hallucination: {n_gen_result[i]['hallucination']}")
            print(f"Correct: {n_gen_result[i]['correct']}")

    return None

# Save individual results to disk
def save_result(result, save_path):
    with open(save_path, mode='a+b') as f:
        for item in result:
            pickle.dump(item, f)


# attribute N-generations
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate multiturn attributes.")
    parser.add_argument("--start", type=int, default=0, help="Start index for the dataset.")
    parser.add_argument("--end", type=int, default=100, help="End index for the dataset.")
    parser.add_argument("--n", type=int, default=3, help="Number of turns to generate.")
    parser.add_argument("--debug", default = False,action="store_true", help="Enable debug mode.")
    parser.add_argument("--data_mining",default = False, action="store_true", help="Enable data mining mode.")
    parser.add_argument("--sensitivity", type=float, default=0.05, help="Sensitivity for the model.")
    args = parser.parse_args()

    #load configurations
    start = args.start
    end = args.end
    n = args.n
    debug = args.debug
    data_mining = args.data_mining
    sensitivity = args.sensitivity
    print(f"Start index: {start}, End index: {end}, n: {n}")

    #load configurations
    if debug: print("Loading configurations...")
    dataset = load_data(start = start, end = end)
    model, tokenizer, embedder = load_model()
    template_name = 'default'
    default_p = template[template_name]

    #load save_path
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(results_dir, f"{start}-{end}_{template_name}_multiturn_{dt_string}.pkl")

    #generate multiturn
    if debug: print("Generating multiturn attributes...")
    p_thresh = 0.5
    for i, (question, aliases) in enumerate(tqdm(dataset)):
        idx = i + start
        if debug: print(f"IDX: Generating multiturn for question {idx}...")
        generate_multiturn_attributes(model, tokenizer, embedder, default_p,question, aliases, n,p_thresh,save_path,idx,sensitivity)
        gc.collect()
