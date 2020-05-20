import jieba
f = open("/Users/chenmingxi/Downloads/weibo_senti_100k/weibo_senti_100k.csv")
lines = f.readlines()
result = open("word_index.txt", 'w')


def stopWords():
    f = open("stopword.txt")
    lines = f.readlines()
    words = []
    for l in lines:
        words.append(l.split('\n')[0])
    return words


def uncommonWords():
    f = open("uncommon.txt")
    lines = f.readlines()
    words = []
    for l in lines:
        words.append(l.split('\n')[0])
    return words


stop = stopWords()
uncommon = uncommonWords()
all_word = []
for i in range(1, len(lines)):
    sentence = lines[i].split('\n')[0][2:]
    words = jieba.cut(sentence)
    for word in words:
        if word in stop or word in uncommon:
            continue
        all_word.append(word)
all_word = list(set(all_word))
print(len(all_word))
for i in range(len(all_word)):
    result.writelines(str(i)+'\t'+all_word[i]+'\n')
result.close()
