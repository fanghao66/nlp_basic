content= ['this is the first document',
        'this is the second second document',
        'and the third one',
        'is this the first document']
import numpy as np
def tf(sentence,token2id):
    p={}
    for word in sentence:
        id = token2id[word]
        p[id] = p.get(id,0)+1
    p={i:j/len(sentence) for i,j in p.items()}
    return list(p.items())

def cal(content_list):
    id2docnum={}
    token2id={}
    max_id=0
    t=0
    for sentence in content_list:
        t+=1
        sentence = list(set(sentence))
        for word in sentence:
            if word not in token2id:
                token2id[word]=max_id
                max_id+=1
            id = token2id[word]
            id2docnum[id]=id2docnum.get(id,0)+1

    return id2docnum,token2id
def idf(content_list,id2docnum):
    idf_={}
    length = len(content_list)
    for id in id2docnum:
        idf_[id] = np.log(length/(id2docnum[id]+1))
    return idf_
def tf_idf(content_list):
    id2docnum,token2id = cal(content_list)
    id2idf = idf(content_list,id2docnum)
    tf_idfs=[]
    for sentence in content_list:
        tf_ = tf(sentence,token2id)
        tf_idf=[]
        for i,j in tf_:
            tf_idf.append((i,j*id2idf[i]))
        tf_idfs.append(tf_idf)
    return tf_idfs

if __name__ == '__main__':
    content_list=[x.split(" ") for x in content]
    print(tf_idf(content_list))
    print()