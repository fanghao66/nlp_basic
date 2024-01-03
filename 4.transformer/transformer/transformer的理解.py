import copy

import numpy as np
import torch
import torch.nn as nn


def qkv_attention_value(q, k, v, mask=False):
    """
    计算attention value
    :param q: [N,T1,E] or [N,h,T1,E]
    :param k: [N,T2,E] or [N,h,T2,E]
    :param v: [N,T2,E] or [N,h,T2,E]
    :param mask: True or False or Tensor
    :return: [N,T1,E] or [N,h,T1,E]
    """
    # 2. 计算q和k之间的相关性->F函数
    k = torch.transpose(k, dim0=-2, dim1=-1)  # [??, T2, E] --> [??, E, T2]
    # matmul([??,T1,E], [??,E,T2])
    scores = torch.matmul(q, k)  # [??,T1,T2]

    if isinstance(mask, bool):
        if mask:
            _shape = scores.shape
            mask = torch.ones((_shape[-2], _shape[-1]))
            mask = torch.triu(mask, diagonal=1) * -10000
            mask = mask[None][None]
        else:
            mask = None
    if mask is not None:
        scores = scores + mask

    # 3. 转换为权重
    alpha = torch.softmax(scores, dim=-1)  # [??,T1,T2]

    # 4. 值的合并
    # matmul([??,T1,T2], [??,T2,E])
    v = torch.matmul(alpha, v)  # [??,T1,E]
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
        v = qkv_attention_value(q, k, v, attention_mask)

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

    def forward(self, q, encoder_k, encoder_v, **kwargs):
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
        v = qkv_attention_value(q, k, v, mask=False)

        # 5. 输出
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [n,h,t,v] --> [n,t,h,v]
        n, t, _, _ = v.shape
        v = torch.reshape(v, shape=(n, t, -1))  # [n,t,h,v] -> [n,t,e]
        v = self.wo(v)  # 多个头之间的特征组合合并
        return v


class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, **kwargs):
        return self.ffn(x)


class ResidualsNorm(nn.Module):
    def __init__(self, block, hidden_size):
        super(ResidualsNorm, self).__init__()
        self.block = block
        self.norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        z = self.block(x, **kwargs)
        z = self.relu(x + z)
        z = self.norm(z)
        return z


class TransformerEncoderLayers(nn.Module):
    def __init__(self, hidden_size, num_header, encoder_layers):
        super(TransformerEncoderLayers, self).__init__()

        layers = []
        for i in range(encoder_layers):
            layer = [
                ResidualsNorm(
                    block=MultiHeadSelfAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=FFN(hidden_size=hidden_size),
                    hidden_size=hidden_size
                )
            ]
            layers.extend(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attention_mask):  # [N,T,E] [N,1,T]
        attention_mask_ = torch.unsqueeze(attention_mask, dim=1)  # 增加header维度
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask_)
        attention_mask = (torch.permute(attention_mask, dims=(0, 2, 1)) >= 0.0).to(x.dtype)  # [N,1,T] -> [N,T,1]
        return x * attention_mask


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_header, max_seq_length, encoder_layers):
        super(TransformerEncoder, self).__init__()

        self.input_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.position_emb = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=hidden_size)
        self.layers = TransformerEncoderLayers(hidden_size, num_header, encoder_layers)

    def forward(self, input_token_ids, input_position_ids, input_mask):
        """
        前向过程
        :param input_token_ids: [N,T] long类型的token id
        :param input_position_ids: [N,T] long类型的位置id
        :param input_mask: [N,1,T] float类型的mask矩阵
        :return:
        """
        # 1. 获取token的embedding
        inp_embedding = self.input_emb(input_token_ids)  # [N,T,E]

        # 2. 获取位置embedding
        position_embedding = self.position_emb(input_position_ids)

        # 3. 合并embedding
        emd = inp_embedding + position_embedding

        # 4. 输入到attention提取特征
        feat_emd = self.layers(emd, attention_mask=input_mask)

        return feat_emd


class TransformerDecoderLayers(nn.Module):
    def __init__(self, hidden_size, num_header, decoder_layers):
        super(TransformerDecoderLayers, self).__init__()

        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)

        layers = []
        for i in range(decoder_layers):
            layer = [
                ResidualsNorm(
                    block=MultiHeadSelfAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=MultiHeadEncoderDecoderAttention(hidden_size=hidden_size, num_header=num_header),
                    hidden_size=hidden_size
                ),
                ResidualsNorm(
                    block=FFN(hidden_size=hidden_size),
                    hidden_size=hidden_size
                )
            ]
            layers.extend(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, encoder_outputs=None, attention_mask=None):
        """
        :param x: [N,T2,E]
        :param encoder_outputs: [N,T1,E]
        :param attention_mask: [N,T2,T2]
        :return:
        """
        attention_mask = torch.unsqueeze(attention_mask, dim=1)  # 增加header维度 [N,T2,T2] -> [N,1,T2,T2]
        k = self.wk(encoder_outputs)  # [N,T1,E]
        v = self.wv(encoder_outputs)  # [N,T1,E]

        for layer in self.layers:
            x = layer(
                x,
                encoder_k=k, encoder_v=v,
                attention_mask=attention_mask
            )
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_header, max_seq_length, decoder_layers):
        super(TransformerDecoder, self).__init__()

        self.input_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.position_emb = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=hidden_size)
        self.layers = TransformerDecoderLayers(hidden_size, num_header, decoder_layers)

    def forward(self, input_token_ids, input_position_ids, input_mask, encoder_outputs):
        """
        前向过程
        :param input_token_ids: [N,T] long类型的token id
        :param input_position_ids: [N,T] long类型的位置id
        :param input_mask: [N,T,T] float类型的mask矩阵
        :param encoder_outputs: [N,T1,E] 编码器的输出状态信息
        :param encoder_attention_mask: [N,T1,T1] 编码器的输入mask信息
        :return:
        """
        if self.training:
            # 1. 获取token的embedding
            inp_embedding = self.input_emb(input_token_ids)  # [N,T,E]

            # 2. 获取位置embedding
            position_embedding = self.position_emb(input_position_ids)

            # 3. 合并embedding
            emd = inp_embedding + position_embedding

            # 4. 输入到attention提取特征
            feat_emd = self.layers(
                emd, encoder_outputs=encoder_outputs,
                attention_mask=input_mask
            )

            return feat_emd
        else:
            raise ValueError("当前模拟代码不实现推理过程，仅实现training过程")


class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, hidden_size, num_header, max_seq_length, encoder_layers,
                 decoder_layers):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(vocab_size=encoder_vocab_size, hidden_size=hidden_size,
                                          num_header=num_header, max_seq_length=max_seq_length,
                                          encoder_layers=encoder_layers)
        self.decoder = TransformerDecoder(vocab_size=decoder_vocab_size, hidden_size=hidden_size,
                                          num_header=num_header, max_seq_length=max_seq_length,
                                          decoder_layers=decoder_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, decoder_vocab_size)
        )

    def forward(self,
                encoder_input_ids, encoder_input_position_ids, encoder_input_mask,
                decoder_input_ids, decoder_input_position_ids, decoder_input_mask):
        if self.training:
            encoder_outputs = self.encoder(encoder_input_ids, encoder_input_position_ids, encoder_input_mask)

            decoder_outputs = self.decoder(
                input_token_ids=decoder_input_ids,
                input_position_ids=decoder_input_position_ids,
                input_mask=decoder_input_mask,
                encoder_outputs=encoder_outputs
            )

            # 决策输出
            scores = self.output_proj(decoder_outputs)
            return scores


def t0():
    encoder = TransformerEncoder(vocab_size=1000, hidden_size=512, num_header=8, max_seq_length=1024, encoder_layers=6)
    decoder = TransformerDecoder(vocab_size=1000, hidden_size=512, num_header=8, max_seq_length=1024, decoder_layers=6)

    input_token_ids = torch.tensor([
        [100, 102, 108, 253, 125],  # 第一个样本实际长度为5
        [254, 125, 106, 0, 0]  # 第二个样本实际长度为3
    ])
    input_position_ids = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ])
    input_mask = torch.tensor([
        [
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0, -10000.0, -10000.0]
        ],
    ])

    input_decoder_token_ids = torch.tensor([
        [251, 235, 124, 321, 25, 68],
        [351, 235, 126, 253, 0, 0]
    ])
    input_decoder_position_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5]
    ])
    input_decoder_mask = torch.tensor([
        [
            [0.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        [
            [0.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, -10000.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, -10000.0, -10000.0, -10000.0],
            [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, -10000.0, 0.0, -10000.0],
            [-10000.0, -10000.0, -10000.0, -10000.0, -10000.0, 0.0]
        ],
    ])

    encoder_outputs = encoder(input_token_ids, input_position_ids, input_mask)
    print(encoder_outputs.shape)

    decoder_outputs = decoder(
        input_token_ids=input_decoder_token_ids,
        input_position_ids=input_decoder_position_ids,
        input_mask=input_decoder_mask,
        encoder_outputs=encoder_outputs
    )
    print(decoder_outputs.shape)


def t1():
    net = Transformer(1000, 1300, 128, 4, 512, 6, 6)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 训练过程模拟 --> 英文到中文的翻译
    # 样本1: He likes to eat apples --> 他 喜 欢 吃 苹 果
    # 样本2: She likes dancing --> 她 喜 欢 跳 舞
    en_word_2_id = {
        '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<EOS>': 3,
        'He': 9, 'likes': 10,
        'to': 4, 'eat': 5,
        'apples': 6, 'She': 7,
        'dancing': 8

    }
    zh_word_2_id = {
        '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<EOS>': 3,
        '他': 11, '喜': 12,
        '欢': 4, '吃': 5,
        '苹': 6, '果': 7,
        '她': 8, '跳': 9,
        '舞': 10
    }
    x = [
        ['He', 'likes', 'to', 'eat', 'apples'],
        ['She', 'likes', 'dancing']
    ]
    y = [
        ['他', '喜', '欢', '吃', '苹', '果'],
        ['她', '喜', '欢', '跳', '舞']
    ]

    # 将编码器输入x进行转换: word2id、填充、mask的构建
    n = len(x)
    t = int(max(map(len, x)))
    # encoder_input_ids, encoder_input_position_ids, encoder_input_mask,
    encoder_input_ids = []
    encoder_input_position_ids = []
    encoder_input_mask = np.ones((n, t))  # [n,t]
    for i, xi in enumerate(x):
        encoder_input_mask[i][:len(xi)] = 0.0
        input_ids = []
        for xw in xi:
            input_ids.append(en_word_2_id.get(xw, en_word_2_id['<UNK>']))
        if len(input_ids) != t:
            input_ids.extend([en_word_2_id['<PAD>']] * (t - len(input_ids)))
        encoder_input_ids.append(input_ids)
        encoder_input_position_ids.append(list(range(t)))
    encoder_input_mask = encoder_input_mask * -10000.0
    encoder_input_ids = torch.tensor(encoder_input_ids, dtype=torch.long)
    encoder_input_position_ids = torch.tensor(encoder_input_position_ids, dtype=torch.long)
    encoder_input_mask = torch.tensor(encoder_input_mask, dtype=torch.float32)
    encoder_input_mask = torch.unsqueeze(encoder_input_mask, dim=1)

    # 将解码器输入y进行转换: word2id、填充、mask的构建
    # decoder_input_ids, decoder_input_position_ids, decoder_input_mask
    decoder_label_ids = []
    decoder_input_ids = []
    decoder_input_position_ids = []
    t = int(max(map(len, y))) + 1
    decoder_loss_mask = np.zeros((n, t))
    decoder_input_mask = np.ones((n, t, t))
    decoder_input_mask = np.triu(decoder_input_mask, k=1)
    for i, yi in enumerate(y):
        decoder_loss_mask[i][:len(yi) + 1] = 1.0
        decoder_input_mask[i][:, len(yi) + 1:] = 1

        input_ids = [zh_word_2_id['<START>']]
        for xw in yi:
            input_ids.append(zh_word_2_id.get(xw, en_word_2_id['<UNK>']))
        input_ids.append(zh_word_2_id['<EOS>'])
        label_ids = []
        label_ids.extend(input_ids[1:])
        label_ids.append(zh_word_2_id['<PAD>'])
        if len(input_ids) < t:
            pad_ids = [zh_word_2_id['<PAD>']] * (t - len(input_ids))
            input_ids.extend(pad_ids)
            label_ids.extend(pad_ids)
        elif len(input_ids) > t:
            input_ids = input_ids[:-1]
            label_ids = label_ids[:-1]
        decoder_input_ids.append(input_ids)
        decoder_label_ids.append(label_ids)
        decoder_input_position_ids.append(list(range(t)))
    decoder_input_mask = decoder_input_mask * -10000.0
    decoder_label_ids = torch.tensor(decoder_label_ids, dtype=torch.long)  # [2,7]
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
    decoder_input_position_ids = torch.tensor(decoder_input_position_ids, dtype=torch.long)
    decoder_input_mask = torch.tensor(decoder_input_mask, dtype=torch.float32)
    decoder_loss_mask = torch.tensor(decoder_loss_mask, dtype=torch.float32) # [2,7]

    r = net(
        encoder_input_ids, encoder_input_position_ids, encoder_input_mask,
        decoder_input_ids, decoder_input_position_ids, decoder_input_mask
    )  # [2, 7, 1300]
    print(r.shape)
    loss = loss_fn(torch.permute(r, dims=(0, 2, 1)), decoder_label_ids)  # [2,7]
    loss = loss * decoder_loss_mask
    loss = loss.mean()
    print(loss.shape)


if __name__ == '__main__':
    t1()

