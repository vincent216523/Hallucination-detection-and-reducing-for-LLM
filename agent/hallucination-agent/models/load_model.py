import os

import numpy as np
import torch
from keras.models import load_model
from xgboost import XGBClassifier

import models.classifer_softmax as sf
from models.classifer_softmax import SoftMaxClassifier
from models.ig import RNNHallucinationClassifier

model_dir = os.path.dirname(__file__)
trained_dir = os.path.join(model_dir, 'trained')

#device
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def load(model_type):
    """
    Load a Keras model from the specified path.
    """
    if model_type == 'activation_MLP':
        model_path = os.path.join(trained_dir, 'activations_score_model_MLP_0.75.keras')
        return load_model(model_path)
    elif model_type == 'activation_XGB':
        model_path = os.path.join(trained_dir, 'activations_score_XGBmodel_0.75.json')
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_path)
        return loaded_model
    elif model_type == 'attention_MLP':
        model_path = os.path.join(trained_dir, 'attention_score_model_MLP_0.68.keras')
        return load_model(model_path)
    elif model_type == 'attention_XGB':
        model_path = os.path.join(trained_dir, 'attention_score_XGBmodel_0.67.json')
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_path)
        return loaded_model
    elif model_type == 'RNN':
        model_path = os.path.join(trained_dir, 'IG_RNN_classifier.pth')
        #model = RNNHallucinationClassifier().to(device)
        IG_rnn_model = torch.load(model_path, weights_only=False).to(device)
        IG_rnn_model.eval()
        return IG_rnn_model
    elif model_type == 'softmax':
        model_path = os.path.join(trained_dir, 'softmax_classifier.pth')
        #model = SoftMaxClassifier(sf.input_size).to(device)
        softmax_model = torch.load(model_path, weights_only=False).to(device)
        softmax_model.eval()
        return softmax_model

