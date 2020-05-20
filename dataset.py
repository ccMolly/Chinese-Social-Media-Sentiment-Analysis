import torch
from torch.utils.data import *
import jieba
import numpy as np


class corpusDataset(Dataset):
    def __init__(self, trainData, wordIndex, maxLen):
        self.sentences = list()
        self.labels = list()
        self.maxLen = maxLen
        self.pair_list = open(trainData).readlines()
        for i in range(1, len(self.pair_list)):
            pair = self.pair_list[i]
            sentence = pair.split('\n')[0][2:]
            label = int(pair[0])
            self.sentences.append(sentence)
            self.labels.append(label)
        self.wor2ind = {}
        for pair in open(wordIndex).readlines():
            index, word = pair.split('\n')[0].split('\t')
            # 0 means padding
            self.wor2ind[word] = int(index) + 1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = []
        words = jieba.cut(self.sentences[index])
        for word in words:
            if word in self.wor2ind and len(sentence) < self.maxLen:
                sentence.append(self.wor2ind[word])
        length = len(sentence)
        if length < self.maxLen:
            sentence.extend([0] * (self.maxLen - length))
        return np.array(sentence), self.labels[index]


# ds_train = corpusDataset("train.txt","word_index.txt", 20)
#
# train_loader = torch.utils.data.DataLoader(ds_train,
#                                            batch_size=5,
#                                            shuffle=True,
#                                            sampler=None)
# for i, (sentence, label) in enumerate(train_loader, 1):
#     print(torch.LongTensor(sentence).shape)
#     print(torch.LongTensor(sentence))
#     print(torch.LongTensor(label).shape)
#     print(torch.LongTensor(label))
#     print("=============")