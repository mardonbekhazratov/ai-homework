import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.randn(self.in_features, self.out_features))
        self.b = nn.Parameter(torch.randn(self.out_features))
    
    def forward(self, x):
        # X shape: B, in_f
        x = x @ self.W + self.b
        return x
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.W = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim))

    def forward(self, x):
        return self.W[x]

config = {
    "context_length": 5,
    "vocab_size": 1000,
    "hidden_size": 50,
    "n_emb": 32
}

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.context_length = config["context_length"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.n_emb = config["n_emb"]
        
        self.E = Embedding(self.vocab_size, self.n_emb)

        self.flat = nn.Flatten()
        self.fc = Linear(self.context_length * self.n_emb, self.hidden_size)
        self.tanh = nn.Tanh()
        self.out = Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        
        x = self.E(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.tanh(x)

        return self.out(x)