import numpy as np

class FNN: # Feedforward Neural Network
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

    def fit(self, X, y):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.Ws = []
        self.bs = []

        for size1, size2 in zip(layer_sizes, layer_sizes[1:]):
            W = np.random.randn(size1, size2)
            b = np.random.randn(size2)
            self.Ws.append(W)
            self.bs.append(b)

        # and we have to fit them ...

    def predict(self, X):
        h_in = X
        for W, b in zip(self.Ws, self.bs):
            z = h_in @ W + b
            h_out = self.heaviside(z)
            h_in = h_out

        return h_out
    
    def heaviside(self, z):
        return 1 if z > 0 else -1 if z < 0 else 0

if __name__=="__main__":
    pass