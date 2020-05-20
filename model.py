import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from embedding import getWeight


class textCNN(nn.Module):
    def __init__(self, pretrain=True):
        super(textCNN, self).__init__()
        if pretrain:
            weight = getWeight()
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(201876, 300)
        self.conv1 = nn.Conv2d(1, 16, (4, 300))
        self.conv2 = nn.Conv2d(1, 16, (5, 300))
        self.conv3 = nn.Conv2d(1, 16, (6, 300))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3 * 16, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x1 = F.relu(x1.squeeze(3))
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)

        x2 = self.conv2(x)
        x2 = F.relu(x2.squeeze(3))
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)

        x3 = self.conv3(x)
        x3 = F.relu(x3.squeeze(3))
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class textRNN(nn.Module):
    def __init__(self, pretrain=True):
        super(textRNN, self).__init__()
        if pretrain:
            weight = getWeight()
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(201876, 300)
        self.lstm = nn.LSTM(300, 128, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        logit = F.log_softmax(x, dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class textRCNN(nn.Module):
    def __init__(self, pretrain=True):
        super(textRCNN, self).__init__()
        if pretrain:
            weight = getWeight()
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(201876, 300)
        self.lstm = nn.LSTM(300, 128, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.maxpool = nn.MaxPool1d(20)
        self.fc = nn.Linear(256+300, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        output, h = self.lstm(x)
        x = torch.cat((x, output), 2)
        x = self.maxpool(self.relu(x).permute(0, 2, 1)).squeeze()
        x = self.fc(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        logit = F.log_softmax(x, dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

