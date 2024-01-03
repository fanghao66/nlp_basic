import torch
import torch.nn as nn

class multi_head_attention(nn.Module):
    def __init__(self,hidden_size,num_headers,masked=None):
        super(multi_head_attention, self).__init__()
        assert hidden_size%num_headers ==0,f"{hidden_size}无法被{num_headers}整除！"
        self.q_layer = nn.Linear(hidden_size,hidden_size)
        self.k_layer = nn.Linear(hidden_size, hidden_size)
        self.v_layer = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Sequential(nn.Linear(hidden_size, hidden_size),nn.ReLU())

        self.num_headers = num_headers
        self.masked = masked
        self.hidden_size = hidden_size
    def split_(self,X):
        N, T, E = X.shape
        hidden_size_split = int(E/self.num_headers)
        X = torch.reshape(X,(N,T,self.num_headers,hidden_size_split))#[N,T,num_headers,hidden_size_split]
        return torch.permute(X,dims=(0,2,1,3))#[N,num_headers,T,hidden_size_split]
    def forward(self,X):
        q = self.q_layer(X)
        k = self.k_layer(X)
        v = self.v_layer(X)

        q = self.split_(q)#[N,num_headers,T,hidden_size_split]
        k = self.split_(k)#[N,num_headers,T,hidden_size_split]
        v = self.split_(v)#[N,num_headers,T,hidden_size_split]

        k = torch.permute(k,dims=(0,1,3,2))
        #1.计算k和q的相似度
        sim_ = torch.matmul(q,k)#[N,num_headers,T,T]
        #2.计算权重weight_
        if self.masked:
            masked = torch.ones(size=(sim_.shape[-1],sim_.shape[-1]))
            masked = torch.triu(masked,diagonal=1)
            masked = masked*-10000#[T,T]
            masked = torch.reshape(masked,shape=(1,1,sim_.shape[-1],sim_.shape[-1]))
            sim_ = masked + sim_
        weight_ = torch.softmax(sim_,dim=-1)#[N,num_headers,T,T]
        #3.计算multi head Attention
        h_attention = torch.matmul(weight_,v)#[N,num_headers,T,hidden_size_split]
        n,_,t,_ = h_attention.shape
        #4.计算attention
        h_attention=torch.permute(h_attention,(0,2,1,3))
        h_attention = torch.reshape(h_attention,(n,t,-1))
        attention = self.wo(h_attention)
        return attention
def t0():
    token_id = torch.tensor([
        [1, 3, 5],  # 表示一个样本，三个时刻
        [1, 6, 3],  # 表示一个样本，三个时刻
        [2, 3, 1],
        [5, 1, 2],
        [6, 1, 2]
    ])

    # 静态特征向量提取 Word2Vec EmbeddingLayer
    emb_layer = nn.Embedding(num_embeddings=10, embedding_dim=8)
    x1 = emb_layer(token_id)  # [2,3,4]
    print(x1[0][0])  # 第一个样本的第一个token对应的向量
    print(x1[1][0])  # 第二个样本的第一个token对应的向量
    print("=" * 100)

    att = multi_head_attention(hidden_size=8, num_headers=2,masked=True)
    x3 = att(x1)
    print(x3[0][0])  # 第一个样本的第一个token对应的向量
    print(x3[1][0])  # 第二个样本的第一个token对应的向量
    print(x3[1][1])  # 第二个样本的第二个token对应的向量


if __name__ == '__main__':
    t0()