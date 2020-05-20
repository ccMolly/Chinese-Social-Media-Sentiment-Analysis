# f = open("/Users/chenmingxi/Downloads/sgns.weibo.word")
# lines = f.readlines()
# for i in range(1,3):
#     a = lines[i].split(' ')
#     print(len(a))
#     print(a)
from gensim.models.keyedvectors import KeyedVectors
import torch


def getWeight():
    wvmodel = KeyedVectors.load_word2vec_format('/Users/chenmingxi/Downloads/sgns.weibo.word', binary=False, unicode_errors='ignore')

    wor2ind = {}
    wor2ind['<pad>'] = 0

    ind2wor = {}
    ind2wor[0] = '<pad>'
    for pair in open('word_index.txt').readlines():
        index, word = pair.split('\n')[0].split('\t')
        # 0 means padding
        wor2ind[word] = int(index) + 1
        ind2wor[int(index) + 1] = word

    weight = torch.zeros(len(wor2ind), 300)

    for i in range(len(wvmodel.index2word)):
        try:
            index = wor2ind[wvmodel.index2word[i]]

        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            ind2wor[wor2ind[wvmodel.index2word[i]]]))

    return weight
