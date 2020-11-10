import numpy as np


class RNN:
    def __init__(self, Wx, Wh, b):  # * 가중치 2개와 편향 1개를 받음(파라미터)
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        h = np.tanh(t)
        self.cache = (x, h_prev, h)
        return h

    def backward(self, dh):
        Wx, Wh, b = self.params
        x, h_prev, h = self.cache
        dt = dh*(1-h**2)
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)  # * *로 리스트의 원소들을 추출하여 RNN에 인수로 전달
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        grads = [0, 0, 0]
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0

        for t in reversed(range(T)):
            dx, dh = self.layers[t].backward(dhs[:, t, :]+dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(self.layers[t].grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs
