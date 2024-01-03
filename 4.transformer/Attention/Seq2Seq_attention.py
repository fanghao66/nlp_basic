import torch
'''Seq2Seq中Attention的实现,q为解码器上一个时刻隐藏层的输出'''
def attention(q,k,v):
    ''' attention
    q:Query from Decoder,shape:[N,E]
    k:Key from Encoder,shape:[N,T,E]
    v:Value from Encoder,shape:[N,T,E]
    :return shape:[N,E]
    '''
    #1.计算k和q之间的相关性
    q=torch.unsqueeze(q,dim=2)#[N,E,1]
    sim_ = torch.matmul(k,q)#[N,T,1]
    #2.根据相似性计算权重值
    weight_ = torch.softmax(sim_,dim=1)#[N,T,1]
    #3.计算Attention值
    attention_=weight_*v#[N,T,E]
    return torch.sum(attention_,dim=1)
