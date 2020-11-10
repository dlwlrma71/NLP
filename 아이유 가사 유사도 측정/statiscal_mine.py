from sklearn.utils.extmath import randomized_svd
from common.util import preprocess, create_co_matrix, ppmi, most_similar
import matplotlib.pyplot as plt
import numpy as np
import sys
import docx2txt
import pandas as pd
sys.path.append('..')
 #* 엑셀로부터 데이터 받아오고, 데이터를 하나로 합친다. 
data = pd.read_excel('아이유.xlsx', sheet_name=1)
data = data['Lyric_Sentence']
data = data + ' '
print(data[0:5])
text = data.sum()

#* 전처리 과정 : ex. 
corpus, word_to_id, id_to_word = preprocess(text)

word_size = len(word_to_id)
C = create_co_matrix(corpus, word_size, window_size=2)
W = ppmi(C, verbose=True)

U, S, V = randomized_svd(W, 500, n_iter=5, random_state=None)
print(U.shape)


def plotdata():
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1], alpha=0.3)
    plt.show()


query_list = ['작은', '문', '아침', '여기']
for query in query_list:
    most_similar(query, word_to_id, id_to_word, U, top=5)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))  # * 각 지점에 이름 매칭

plt.scatter(U[:, 0], U[:, 1], alpha=0.1)  # *plot
plt.show()
