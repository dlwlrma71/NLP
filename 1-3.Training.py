from ch01.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from dataset import spiral
from common.optimizer import SGD
import numpy as np
import sys
sys.path.append('..')

# * 하이퍼파라미터 설정 (300 에폭, 배치 30, 히든 : 10 , lr = 0.1)
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 0.1

# * 데이터 읽기, 모델과 옵티마이저 생성 (spiral.load_data()이용, Twolayernet, SGD)
x, t = spiral.load_data()
print(x[0:3], t[0:3])
model = TwoLayerNet(
    input_size=x.shape[1], hidden_size=hidden_size, output_size=t.shape[1])
optimizer = SGD(lr=lr)

iters_num = len(x)//batch_size
loss_list = []
for epoch in range(max_epoch):
    # * 데이터 뒤섞기 (에폭마다, permutation함수 사용)
    index = np.random.permutation(len(x))
    x = x[index]
    t = t[index]
    total_loss, count = 0, 0
# * 기울기를 구해 매개변수 갱신
    for iter in range(iters_num):
        batch_x = x[batch_size*iter:batch_size*(iter+1)]
        batch_t = t[batch_size*iter:batch_size*(iter+1)]
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        count += 1
        total_loss += loss

        if (iter+1) % 10 == 0:
            average_loss = total_loss/count
            print("에폭:%d\t반복:%d/%d\t평균loss:%.2f" %
                  (epoch+1, iter+1, iters_num, average_loss))
            loss_list.append(average_loss)
            total_loss, count = 0, 0

# * 정기적으로 학습경과 출력
