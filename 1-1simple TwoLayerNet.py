import numpy as np

# * sigmoid


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1/(1+np.exp(x))

# * Affine


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]  # params 리스트에 W,b를 받아서 저장

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

# * twolayerNet


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # *가중치 초기화
        W1, W2 = np.random.randn(I, H), np.random.randn(H, O)
        b1, b2 = np.random.randn(H), np.random.randn(O)
        # *계층 생성
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        # * params리스트에 모으기
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    # * predict

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
