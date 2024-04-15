import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Position(nn.Module):
    def __init__(self, batch_size, seq_length, input_size):
        super(Position, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.batch_size = batch_size
         
    def get_position(self):
        d = self.input_size
        seq_length = self.seq_length
        batch_size = self.batch_size

        p = torch.empty((batch_size, seq_length, d), dtype=torch.float32)
        w = torch.exp(torch.arange(0., d, 2) * (-math.log(10000.0) / d))
        # w: (embedding_size/2)

        position = torch.arange(0, seq_length).unsqueeze(1)
        # position: (seq_length, 1)

        p[:,:, 0::2] = torch.sin(position*w)
        p[:,:, 1::2] = torch.cos(position*w)
        # 用矩阵运算代替循环更快
        
        return p
        
class Embedding(nn.Module):
    def __init__(self, batch_size, seq_length, input_size):
        super(Embedding, self).__init__()
        self.vocab_size = input_size
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.embedding = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        embed_x = self.embedding(x)
        return embed_x


class MultiAttention(nn.Module):
    def __init__(self, hidden_size, input_size, num_heads):
        super(MultiAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        
        # Q, K, V，Q = inputs * W_Q
        self.W_Q = nn.Linear(input_size, hidden_size)
        # k, v的维度需要一致；为了计算方便，将此处Q的维度也设为input_s
        self.W_K = nn.Linear(input_size, hidden_size)
        self.W_V = nn.Linear(input_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs, mask = False):

        batch_size, seq_length, _ = inputs.shape
        hidden_size = self.hidden_size
        num_heads = self.num_heads
        
        Q = self.W_Q(inputs)
        V = self.W_V(inputs)
        K = self.W_K(inputs)

        # multi-head
        Q = Q.view(batch_size, -1, seq_length, hidden_size//num_heads)
        V = V.view(batch_size, -1, seq_length, hidden_size//num_heads)
        K = K.view(batch_size, -1, seq_length, hidden_size//num_heads).transpose(-1,-2)

        # scaled scores
        e = torch.matmul(Q, K)
        e = e/ math.sqrt(hidden_size)

        # mask
        if mask:
            # e.masked_fill_(torch.triu(torch.ones_like(e), diagonal=1) == 1, float("-inf"))
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            a = e.masked_fill_(mask, -1e10)
        else:
            # attention_distribution
            softmax = self.softmax
            a = softmax(e)

        # output
        o = torch.matmul(a,V)
        o = o.view(batch_size, seq_length, hidden_size)
            
        return o

class Weights(nn.Module):
    def __init__(self, hidden_size, input_size, num_heads):
        super(Weights, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        
        # Q, K, V，Q = inputs * W_Q
        self.W_Q = nn.Linear(input_size, hidden_size)
        # k, v的维度需要一致；为了计算方便，将此处Q的维度也设为input_s
        self.W_K = nn.Linear(input_size, hidden_size)
        self.W_V = nn.Linear(input_size, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs):

        batch_size, seq_length, _ = inputs.shape
        hidden_size = self.hidden_size
        num_heads = self.num_heads
        
        Q = self.W_Q(inputs)
        V = self.W_V(inputs)
        K = self.W_K(inputs)

        # multi-head
        Q = Q.view(batch_size, -1, seq_length, hidden_size//num_heads)
        V = V.view(batch_size, -1, seq_length, hidden_size//num_heads)
        K = K.view(batch_size, -1, seq_length, hidden_size//num_heads).transpose(-1,-2)
            
        return Q, K, V
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, Q, K, V, mask = False):

        batch_size, _, seq_length, _ = Q.shape
        hidden_size = self.hidden_size

        # scaled scores
        e = torch.matmul(Q, K)/ math.sqrt(hidden_size)

        # mask
        if mask:
            e.masked_fill_(torch.triu(torch.ones_like(e), diagonal=1) == 1, float("-inf"))

        # attention_distribution
        softmax = self.softmax
        a = softmax(e)

        # output
        o = torch.matmul(a,V)
        o = o.view(batch_size, seq_length, hidden_size)
            
        return o

class FeedForward(nn.Module):
    def __init__(self, attn_size, ff_size):
        super(FeedForward, self).__init__()
        self.attn_size = attn_size
        self.ff_size = ff_size
        self.linear_2ff = nn.Linear(attn_size, ff_size)

    def forward(self, x):
        x = self.linear_2ff(x)
        x = F.relu(x)
        
        return x
    
class Outputs(nn.Module):
    def __init__(self, ff_size, vocab_size):
        super(Outputs, self).__init__()
        self.linear = nn.Linear(ff_size, vocab_size)
        self.softmax = nn.Softmax(dim =-1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x