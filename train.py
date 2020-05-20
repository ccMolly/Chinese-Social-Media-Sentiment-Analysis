import torch
import torch.nn as nn
from torch.utils.data import *
from dataset import corpusDataset
from model import textCNN, textRNN, textRCNN

ds_train = corpusDataset("train.txt","word_index.txt", 20)

train_loader = torch.utils.data.DataLoader(ds_train,
                                           batch_size=64,
                                           shuffle=True,
                                           sampler=None)

ds_test = corpusDataset("test.txt","word_index.txt", 20)

test_loader = torch.utils.data.DataLoader(ds_test,
                                           batch_size=1,
                                           shuffle=True,
                                           sampler=None)

model = textRNN(pretrain=True)
model.init_weight()
# checkpoint = torch.load('checkpoints_lstm/crnn_0_1000.pth')
# model.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()


def validate():
    correct = 0
    length = len(test_loader)
    for i, (sentence, label) in enumerate(test_loader, 1):
        sentence = torch.LongTensor(sentence)
        label = torch.LongTensor(label)
        predict = model(sentence)
        result = torch.argmax(predict, dim=1)
        if result[0] == label[0]:
            correct += 1
    print("==============================")
    print("ACC:", correct/length)
    print("==============================")


for epoch in range(0,2):
    for i, (sentence, label) in enumerate(train_loader, 1):
        batch_size = label.shape[0]
        sentence = torch.LongTensor(sentence)
        label = torch.LongTensor(label)
        predict = model(sentence)
        loss = criterion(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result = torch.argmax(predict,dim=1)
        correct = 0
        for j in range(batch_size):
            if result[j] == label[j]:
                correct+=1
        if i % 50 == 0:
            print("Iteration:", i, "Loss:", loss.data[0], "Correct:", correct)
        if i > 0 and i % 100 == 0:
            validate()
         if i >0 and i % 300 == 0:
             torch.save({'epoch': epoch,
                          'state_dict': model.state_dict()},
                         '{0}/crnn_{1}_{2}.pth'.format('checkpoints_rcnn', epoch, i))

