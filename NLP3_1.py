from common.util import convert_one_hot
import numpy as np
import sys
from common.layers import MatMul, SoftmaxWithLoss
from common.util import preprocess
from common.trainer import Trainer
from common.optimizer import Adam
from ch03.simple_cbow import SimpleCBOW
sys.path.append('..')

text = 'The only way to this is to enjoy it'
corpus, word_to_id, id_to_word = preprocess(text)

# * contexts, target 데이터 생성


def create_contexts_target(corpus, window_size=1):
    contexts = np.zeros(((len(corpus)-2*window_size),
                         2*window_size), dtype=np.int32)
    for i in range(len(contexts)):
        contexts[i] = [corpus[j]
                       for j in range(i, i+2*window_size+1) if j != i+1]
    target = corpus[window_size:-window_size]
    return np.array(contexts), np.array(target)


contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts, target)
vocab_size = len(word_to_id)
# * 주어진 함수로 one-hot vector 생성
target, contexts = convert_one_hot(
    target, vocab_size), convert_one_hot(contexts, vocab_size)

# ! 간단한 신경망 구현


class simpleCBOWmy:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = np.random.randn(V, H).astype('f')
        W_out = np.random.randn(H, V).astype('f')
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        # * 인스턴스 변수에 단어 분산 표현 저장
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0+h1)*0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)  # * 이것만으로도 grads 리스트의 기울기가 갱신된다.
        return None


class myskipgram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01*np.random.randn(V, H).astype('f')
        W_out = 0.01*np.random.randn(H, V).astype('f')
        self.inlayer = MatMul(W_in)
        self.outlayer = MatMul(W_out)
        self.losslayer1 = SoftmaxWithLoss()
        self.losslayer2 = SoftmaxWithLoss()
        layers = [self.inlayer, self.outlayer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        s = self.inlayer.forward(target)
        h = self.outlayer.forward(s)
        loss1 = self.losslayer1.forward(h, contexts[:, 0])
        loss2 = self.losslayer2.forward(h, contexts[:, 1])
        loss = loss1+loss2
        return loss

    def backward(self, dout=1):
        dl1 = self.losslayer1.backward(dout)
        dl2 = self.losslayer2.backward(dout)
        ds = dl1+dl2
        dh = self.outlayer.backward(ds)
        self.inlayer.backward(dh)
        return None


# * 모델 훈련 (Adam) input: hiddensize, batchsize, epoch
hidden_size = 5
batch_size = 3
max_epoch = 1000

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

# * skig gram 훈련
hs = 5
bs = 3
m_epoch = 1000
modelA = myskipgram(vocab_size, hs)
optimizerA = Adam()
trainerA = Trainer(modelA, optimizerA)
trainerA.fit(contexts, target, m_epoch, bs)
trainerA.plot()

wordvecs = modelA.word_vecs
for word_id, word in id_to_word.items():
    print(word, wordvecs[word_id])
