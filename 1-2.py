from common.layers import SoftmaxWithLoss, Sigmoid
import os
import numpy as np
import sys
sys.path.append("/Users/dlwlrma71/Desktop/deep-learning-from-scratch-2-master")
sys.path.append(os.listdir(
    "/Users/dlwlrma71/Desktop/deep-learning-from-scratch-2-master"))
#! Matmul 계층 : W를 self.params에 저장, dW를 self.grads에 저장 out, dx는 return!


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.X = None

    def forward(self, X, b):
        W, = self.params
        self.X = X
        out = np.matmul(X, W)
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.X.T, dout)
        self.grads[0][...] = dW
        return dx

# * sigmoid 계층


# class Sigmoid():
#     def __init__(self):
#         self.params, self.grads = [], []
#         self.out = None

#     def forward(self, x):
#         self.out = 1/(1+np.exp(-x))
#         return self.out

#     def backward(self, dout):
#         dx = dout*self.out*(1-self.out)
#         return dx

#! Affine계층


class Affine():
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W = self.params[0]
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

# * SGD계층


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]

#! 신경망 구현 I,H,O: hyperparameter


class TwoLayerNet:
    def __init__(self, input_num, hidden_num, output_num):
        I, H, O = input_num, hidden_num, output_num
        # *가중치, 편향 초기화
        W1, W2 = np.random.randn(I, H), np.random.randn(H, O)
        b1, b2 = np.zeros(H), np.zeros(O)
        # *계층 생성
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        self.last_layer = SoftmaxWithLoss()
        # *모든 가중치, 기울기를 리스트에 모으기
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        # *predict

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        # *forward(loss계산)

    def forward(self, x, t):
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        return loss
        # *backward

    def backward(self, dout):
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
