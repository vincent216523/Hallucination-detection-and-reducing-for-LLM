import pickle
from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
batch_size = 128
dropout_mlp = 0.5
dropout_gru = 0.25
learning_rate = 1e-4
weight_decay = 1e-2
epoch_size = 1000
input_size = 32000
random_state = 21
inference_results = list(Path("./data/").rglob("*.pickle"))


class SoftMaxClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=dropout_mlp):
        super().__init__()
        self.dropout = dropout
        self.hidden_layer1_size = 128
        self.output_layer_size = 2
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_shape, self.hidden_layer1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            #torch.nn.Linear(2048, 256),
            #torch.nn.ReLU(),
            #torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_layer1_size, self.output_layer_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def gen_classifier_roc(inputs, labels, classifier_model):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels.astype(int),
                                                        test_size = 0.2, random_state=random_state)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    classifier_model.train()
    for epoch in enumerate(tqdm(range(epoch_size))):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()

    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
        return roc_auc_score(y_test.cpu(), pred[:,1].cpu()), (prediction_classes.numpy()==y_test.cpu().numpy()).mean()

softmax_model = SoftMaxClassifier(input_size).to(device)
# softmax_model = torch.load('softmax_classifier.pth', weights_only=False).to(device)  # for retrain


df_list = list()
for results_file in inference_results:
    with open(results_file, "rb") as infile:
        df_list.append(pd.DataFrame(pickle.loads(infile.read())))
results = pd.concat(df_list, ignore_index=True)

first_logits = np.stack([sp.special.softmax(i[j]) for i, j in zip(results['logits'], results['start_pos'])])
labels = np.array(results['correct'])
first_logits_roc, first_logits_acc = gen_classifier_roc(first_logits, labels, softmax_model)
print(f"first_logits_roc:{first_logits_roc}, first_logits_acc:{first_logits_acc}")
torch.save(softmax_model, 'softmax_classifier.pth')
