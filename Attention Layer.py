import numpy as np


#! Attention Decoder 개선 1

#! idea : 기존에는 마지막 은닉벡터 h만 사용했지만, hs전부를 활용, 대응관계를 학습시키기 위함임
#! hs의 특정 가중치 a를 곱해서, 가중합 벡터 c를 얻는다. 해당 클래스에는 학습시킬 파라미터는 존재하지 않는다.
#! 즉, hs중 특정 벡터를 선택하는 작업을 c라는 가중합으로 대체하는 것으로 볼 수 있다.

N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
# * a: 각 단어 벡터별 가중치를 의미하는 벡터
a = np.random.randn(N, T)
ar = a.reshape(N, T, 1).repeat(H, axis=2)
t = hs*ar
print(t.shape)
c = np.sum(t, axis=1)
print(c.shape)


class WeightSum:
    def __init__(self):
        self.parmas, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs*ar
        c = np.sum(t, axis=1)
        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt*hs
        dhs = dt*ar
        da = np.sum(dar, axis=2)
        return dhs, da

#! Attention Decoder 개선 2 - how to get a?

#! Decoder LSTM의 각 출력 h에 hs를 내적한 다음, softmax


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)  # !따로 구현하진 않음
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.hs = None
        self.Softmax = Softmax()
        self.hr = None

    def forward(self, hs, h):
        self.hs = hs
        N, T, H = hs.shape
        hr = np.reshape(h, (N, 1, H)).repeat(T, axis=1)
        t = self.hr*hs
        s = np.sum(t, axis=2)
        a = Softmax.forward(s)
        self.hr = hr
        return a

    def backward(self, da):
        N, T, H = self.hs.shape
        ds = Softmax.backward(da)
        dt = np.reshape(ds, (N, T, 1)).repeat(H, axis=2)
        dhr = dt*self.hs
        dhs = dt*self.hr
        dh = np.sum(dhr, axis=1)

        return dhs, dh

#! Attention Decoder 개선 3 - 두 layer 합쳐서 Attention layer 구현


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.Attention_weight_layer = AttentionWeight()
        self.Weight_sum_layer = WeightSum()
        #! 가중치 a(N,T)를 나중에 참조할 수 있도록!
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.Attention_weight_layer.forward(hs, h)
        c = self.Weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return c

    def backward(self, dc):
        dhs0, da = self.Weight_sum_layer.backward(dc)
        dhs1, dh = self.Attention_weight_layer.backward(da)
        dhs = dhs0+dhs1
        return dhs, dh

#! 최종적으로 TimeAttention 구현


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weights = None
        self.layers = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.attention_weights.append(layer.attention_weight)
            self.layers.append(layer)

        return out

    def bakward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout)
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
