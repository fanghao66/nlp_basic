content= ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document']
import jieba
def BOW(content):
    token2id={}
    max_id=0
    token2bow=[]
    for sentence in content:
        token = sentence.split(" ")
        freq = {}
        for word in token:
            if word not in token2id:
                token2id[word]=max_id
                max_id+=1
            id = token2id[word]
            freq[id] = freq.get(id, 0) + 1
        token2bow.append(list(freq.items()))
    return token2bow,token2id
print(BOW(content))
print()