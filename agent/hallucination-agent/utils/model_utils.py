import functools
import gc
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from string import Template
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

'''
Thie module was adapted from the original implementation from the llm-hallucination github repo

'''
# Global variables
ig_steps = 64
internal_batch_size = 1

# Model
model_name = "open_llama_7b" #"opt-30b"
layer_number = -1
# hardcode below,for now. Could dig into all models but they take a while to load
model_num_layers = {
    "falcon-40b" : 60,
    "falcon-7b" : 32,
    "open_llama_13b" : 40,
    "open_llama_7b" : 32,
    "opt-6.7b" : 32,
    "opt-30b" : 48,
}
assert layer_number < model_num_layers[model_name]
coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
model_repos = {
    "falcon-40b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "falcon-7b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "open_llama_13b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "open_llama_7b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "opt-6.7b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
    "opt-30b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj", ),
}

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

def load_model():
    # IO
    data_dir = Path(".") # Where our data files are stored
    model_dir = Path("./.cache/models/") # Cache for huggingface models

    # Model
    model_name = "open_llama_7b" #"opt-30b"
    layer_number = -1

    # hardcode below,for now. Could dig into all models but they take a while to load
    model_num_layers = {
        "open_llama_7b" : 32
    }
    assert layer_number < model_num_layers[model_name]
    coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
    model_repos = {
        "open_llama_7b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    }

    model_loader = LlamaForCausalLM
    token_loader = LlamaTokenizer
    tokenizer = token_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}')
    model = model_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}',
                                         cache_dir=model_dir,
                                         device_map=device,
                                         torch_dtype=torch.bfloat16,
                                         # load_in_4bit=True,
                                         trust_remote_code=True)
    embedder = model.model.embed_tokens
    return model, tokenizer, embedder


def generate_response(x, model, *, max_length, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x, model)
        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == 13 and step > 5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze()


def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits

def answer_trivia(question, targets, model, tokenizer, pbar = False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    max_length = input_ids.shape[1] * 3
    response, logits, start_pos = answer_question(question, model, tokenizer, max_length=max_length, pbar=pbar)
    start_pos = input_ids.shape[-1]
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    for alias in targets:
        if alias.lower() in str_response.lower():
            correct = True
            break
    return response, str_response, logits, start_pos, correct

def answer_question(question, model, tokenizer, *, max_length, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    response, logits = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    return response, logits, input_ids.shape[-1]

def get_start_end_layer(model):
    model_layers = model.model.layers
    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = len(model_layers) if layer_number == -1 else layer_number + 1
    return layer_st, layer_en

def save_fully_connected_hidden(fully_connected_hidden_layers,layer_name, mod, inp, out):
    fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def save_attention_hidden(attention_hidden_layers,layer_name, mod, inp, out):
    attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def collect_fully_connected(fully_connected_hidden_layers,token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    first_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
    return first_activation, final_activation


def collect_attention(attention_hidden_layers,token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    first_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
    return first_activation, final_activation

def get_ig(prompt, forward_func, tokenizer, embedder, model):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    prediction_id = get_next_token(input_ids, model).squeeze()[-1].argmax()
    encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
    ig = IntegratedGradients(forward_func=forward_func)
    attributes = normalize_attributes(
        ig.attribute(
            encoder_input_embeds,
            target=prediction_id,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size
        )
    ).detach().cpu().numpy()
    return attributes

def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        norm = torch.norm(attributes, dim=1)
        attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1

        return attributes

def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        output = model(inputs_embeds=input_, **extra_forward_args)
        return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


def generate_attributes(model, tokenizer, embedder, question,targets):

    # For storing results
    results = defaultdict(list)
    fully_connected_hidden_layers = defaultdict(list)
    attention_hidden_layers = defaultdict(list)
    attention_forward_handles = {}
    fully_connected_forward_handles = {}

    # Regex expressions for model layers
    llama_mlp_regex = re.compile(r".*model.layers.[0-9]+.mlp.up_proj")
    llama_attn_regex = re.compile(r".*model.layers.[0-9]+.self_attn.o_proj")

    # Prepare to save the internal states
    for name, module in model.named_modules():
        if re.match(f'{model_repos[model_name][1]}$', name):
            fully_connected_forward_handles[name] = module.register_forward_hook(
                partial(save_fully_connected_hidden, fully_connected_hidden_layers,name))
        if re.match(f'{model_repos[model_name][2]}$', name):
            attention_forward_handles[name] = module.register_forward_hook(
                partial(save_attention_hidden, attention_hidden_layers,name))

    #forward function init
    forward_func = partial(model_forward, model=model, extra_forward_args={})

    # Generate results
    fully_connected_hidden_layers.clear()
    attention_hidden_layers.clear()

    # Generate response and get atributes
    response, str_response, logits, start_pos, correct = answer_trivia(question, targets, model, tokenizer)
    layer_start, layer_end = get_start_end_layer(model)
    first_fully_connected, final_fully_connected = collect_fully_connected(fully_connected_hidden_layers,start_pos, layer_start, layer_end)
    first_attention, final_attention = collect_attention(attention_hidden_layers,start_pos, layer_start, layer_end)
    attributes_first = get_ig(question, forward_func, tokenizer, embedder, model)

    # Store attributes
    results['question'].append(question)
    results['response'].append(response)
    results['str_response'].append(str_response)
    results['logits'].append(logits.to(torch.float32).cpu().numpy())
    results['start_pos'].append(start_pos)
    results['correct'].append(correct)
    results['first_fully_connected'].append(first_fully_connected)
    results['final_fully_connected'].append(final_fully_connected)
    results['first_attention'].append(first_attention)
    results['final_attention'].append(final_attention)
    results['attributes_first'].append(attributes_first)


    for handle in fully_connected_forward_handles.values():
        handle.remove()
    for handle in attention_forward_handles.values():
        handle.remove()
    del fully_connected_hidden_layers
    del attention_hidden_layers
    gc.collect()

    return results
