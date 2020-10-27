from ch04.negative_sampling_layer import UnigramSampler
from common.layers import SoftmaxWithLoss, SigmoidWithLoss
import sys
import numpy as np
sys.path.append('..')

# * Matmul계층을 대신할 Embedding 계층 구현


class Embedding():
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx  # *추출하는 행의 인덱스를 배열로 저장
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        # *for id, word_id in enumerate(self.idx):
        # *    dW[word_id] += dout[i]
        np.add.at(dW, self.idx, dout)  # *dout을 dW의 idx번째 행에 더해줌
        # * dh의 각 행 값을 idx가 가리키는 장소에 할당/더해줌 (더해주는 것이 더 맞는 표현)
        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout*h
        self.embed.backward(dtarget_W)
        dh = dout*target_W
        return dh

# * 네거티브 샘플링계층 (loss까지!) h부터 loss까지!


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(
            corpus, power, sample_size)  # *negative samples
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        # *negativesamples + 1 positive sample
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]  # * ex. [0,3,1] -> 3
        negative_sample = self.sampler.get_negative_sample(
            target)  # * [0,3,1]이 아닌 negative sample 추출

        # * 맞는 예
        # * target : 맞는 예의 index -> return : h*W[target]
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)  # * [1,1,1]
        loss = self.loss_layers[0].forward(
            score, correct_label)  # * 세개의 loss 더해서 합치는듯?

        # * negative
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            y = self.embed_dot_layers[i+1].forward(h, negative_target)
            loss += self.loss_layers[i+1].forward(y, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for sig, embed in (self.loss_layers, self.embed_dot_layers):
            dscore = sig.backward(dout)
            dh += embed.backward(dscore)
        return dh
