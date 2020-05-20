import jieba
f = open("/Users/chenmingxi/Downloads/weibo_senti_100k/weibo_senti_100k.csv")
lines = f.readlines()
train = open("train.txt", 'w')
test =  open("test.txt", 'w')
length = len(lines)
for i in range(1,int(length*0.1)):
    test.writelines(lines[i])
for i in range(int(length*0.1),int(length*0.9)):
    train.writelines(lines[i])
for i in range(int(length*0.9),length):
    test.writelines(lines[i])
test.close()
train.close()