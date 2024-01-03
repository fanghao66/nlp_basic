'''tansformer 和 Bert中所用的attention算法'''
import torch
import torch.nn as nn
'''
transformer中的attention：
1.基本attention函数
2.MultiHeadSelfAttention
3.MultiHeadSelfAttention
'''
def attention(q,k,v,mask=False):
    '''
    :param q:Query,[N,T1,E] or [N,h,T1,v]
    :param k: Key,[N,T2,E] or [N,h,T2,v]
    :param v: Value,[N,T2,E] or [N,h,T2,v]
    :param mask:bool or [T1,T2]
    :return: output tensor,[N,T,E]
    '''
    _dim = len(q.shape)#3 or 4
    scores = torch.matmul(q,torch.permute(k,(0,1,3,2)))
    if isinstance(mask,bool):
        if mask:
            _shape = scores.shape
            mask = torch.ones(_shape[-2],_shape[-1])
            mask = torch.triu(mask,diagonal=1)*-10000
            if _dim ==4:
                mask = mask[None][None]
                scores += mask
            else:
                mask = mask[None]
                scores += mask
    else:
        scores += mask
    weight_ = torch.softmax(scores,dim=-1)#[??,T1,T2]
    v = torch.matmul(weight_,v)
    return v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_header):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_header == 0, f"header的数目没办法整除:{hidden_size}, {num_header}"

        self.hidden_size = hidden_size  # 就是向量维度大小，也就是E
        self.num_header = num_header  # 头的数目

        self.wq = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wk = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wv = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        )
        self.wo = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        )

    def split(self, vs):
        n, t, e = vs.shape
        vs = torch.reshape(vs, shape=(n, t, self.num_header, e // self.num_header))
        vs = torch.permute(vs, dims=(0, 2, 1, 3))
        return vs

    def forward(self, x, attention_mask=None, **kwargs):
        """
        前向过程
        :param x: [N,T,E] 输入向量
        :param attention_mask: [N,T,T] mask矩阵
        :return: [N,T,E] 输出向量
        """
        # 1. 获取q、k、v
        q = self.wq(x)  # [n,t,e]
        k = self.wk(x)  # [n,t,e]
        v = self.wv(x)  # [n,t,e]
        q = self.split(q)  # [n,t,e] --> [n,h,t,v]  e=h*v h就是head的数目，v就是每个头中self-attention的维度大小
        k = self.split(k)  # [n,t,e] --> [n,h,t,v]  e=h*v
        v = self.split(v)  # [n,t,e] --> [n,h,t,v]  e=h*v

        # 计算attention value
        v = attention(q, k, v, attention_mask)

        # 5. 输出
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [n,h,t,v] --> [n,t,h,v]
        n, t, _, _ = v.shape
        v = torch.reshape(v, shape=(n, t, -1))  # [n,t,h,v] -> [n,t,e]
        v = self.wo(v)  # 多个头之间的特征组合合并
        return v

class MultiHeadEncoderDecoderAttention(nn.Module):
    def __init__(self, hidden_size, num_header):
        super(MultiHeadEncoderDecoderAttention, self).__init__()
        assert hidden_size % num_header == 0, f"header的数目没办法整除:{hidden_size}, {num_header}"

        self.hidden_size = hidden_size  # 就是向量维度大小，也就是E
        self.num_header = num_header  # 头的数目

        self.wo = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        )

    def split(self, vs):
        n, t, e = vs.shape
        vs = torch.reshape(vs, shape=(n, t, self.num_header, e // self.num_header))
        vs = torch.permute(vs, dims=(0, 2, 1, 3))
        return vs

    def forward(self, q, encoder_k, encoder_v, encoder_attention_mask, **kwargs):
        """
        编码器解码器attention
        :param q: [N,T1,E]
        :param encoder_k: [N,T2,E]
        :param encoder_v: [N,T2,E]
        :param encoder_attention_mask: [N,1,T2,T2]
        :return: [N,T1,E]
        """
        q = self.split(q)  # [n,t,e] --> [n,h,t,v]  e=h*v h就是head的数目，v就是每个头中self-attention的维度大小
        k = self.split(encoder_k)  # [n,t,e] --> [n,h,t,v]  e=h*v
        v = self.split(encoder_v)  # [n,t,e] --> [n,h,t,v]  e=h*v

        # 计算attention value
        v = attention(q, k, v, mask=encoder_attention_mask)

        # 5. 输出
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [n,h,t,v] --> [n,t,h,v]
        n, t, _, _ = v.shape
        v = torch.reshape(v, shape=(n, t, -1))  # [n,t,h,v] -> [n,t,e]
        v = self.wo(v)  # 多个头之间的特征组合合并
        return v
