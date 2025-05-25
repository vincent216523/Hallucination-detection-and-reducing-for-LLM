import torch


class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)

    def forward(self, seq):
        gru_out, _ = self.gru(seq)
        return self.linear(gru_out)[-1, -1, :]
