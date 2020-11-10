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

#* 전처리 과정 (NLP폴더에서 직접 구현, 여기서는 불러와서 사용)
corpus, word_to_id, id_to_word = preprocess(text)

word_size = len(word_to_id)
C = create_co_matrix(corpus, word_size, window_size=2)
W = ppmi(C, verbose=True)

#* sklearn SVD사용 
U, S, V = randomized_svd(W, 500, n_iter=5, random_state=None)
print(U.shape)

#* 단어 벡터 1,2번째 핵심벡터만을 이용한 그래프 plot 
def plotdata():
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1], alpha=0.3)
    plt.show()

# * 샘플 몇 개의 단어(query)와 가장 유사한 단어 상위 5개 측정  유사도: 벡터간 cos similarity 사용

query_list = ['편지', '문', '음악', '밤']
for query in query_list:
    most_similar(query, word_to_id, id_to_word, U, top=5)

