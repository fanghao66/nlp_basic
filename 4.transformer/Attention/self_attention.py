'''bert中Self Attention的实现'''
import torch
import torch.nn as nn
class Self_Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Self_Attention, self).__init__()
        self.q_layer=nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.k_layer = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.v_layer = nn.Linear(in_features=hidden_size,out_features=hidden_size)
    def forward(self, X):
        '''
        X: input_tensor,[N,T,E]
        return：output_tensor,[N,T,E]
        '''

        q = self.q_layer(X)#[N,T,E]
        k = self.k_layer(X)#[N,T,E]
        v = self.v_layer(X)#[N,T,E]
        #1.计算q和k之间的相似性
        k = torch.permute(k,dims=(0,2,1))#[N,E,T]
        sim_ = torch.matmul(q,k)#[N,T,T]
        #2.根据相似性计算权重
        weight_=torch.softmax(sim_,dim=2)#[N,T,T]
        #3.计算attention值
        attention_ = torch.matmul(weight_,v)#[N,T,E]
        return attention_

def t0():
    token_id = torch.tensor([
        [1, 3, 5],  # 表示一个样本，三个时刻
        [1, 6, 8]  # 表示一个样本，三个时刻
    ])

    # 静态特征向量提取 Word2Vec EmbeddingLayer
    emb_layer = nn.Embedding(num_embeddings=10, embedding_dim=4)
    x1 = emb_layer(token_id)  # [2,3,4]
    print(x1[0][0])  # 第一个样本的第一个token对应的向量
    print(x1[1][0])  # 第二个样本的第一个token对应的向量
    print("=" * 100)

    # 如何能够让不同文本中的相同token最终对应的特征向量是不同的？ --> 用到文本的整个序列特征
    # 56行到72行代码是有问题的，参数依赖序列长度，是不可以的
    t = x1.shape[1]
    fc_layers = [
        nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        ) for _ in range(t)
    ]
    x2 = []
    for i in range(t):
        x2_ = fc_layers[i](x1)
        x2_ = torch.mean(x2_, dim=1, keepdim=True)
        x2.append(x2_)
    x2 = torch.concat(x2, dim=1)
    print(x2[0][0])  # 第一个样本的第一个token对应的向量
    print(x2[1][0])  # 第二个样本的第一个token对应的向量
    print(x2[1][1])  # 第二个样本的第二个token对应的向量
    print("=" * 100)

    # 基于self-attention的提取
    att = Self_Attention(hidden_size=4)
    x3 = att(x1)
    print(x3[0][0])  # 第一个样本的第一个token对应的向量
    print(x3[1][0])  # 第二个样本的第一个token对应的向量
    print(x3[1][1])  # 第二个样本的第二个token对应的向量


if __name__ == '__main__':
    t0()