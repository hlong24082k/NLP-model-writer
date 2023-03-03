import math
import torch
import torch.nn as nn
import numpy as np

def create_look_head_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask


def create_padding_mask(input):
    padding_mask = (input == 0).float()
    batch_size, seg_len = padding_mask.shape
    padding_mask = padding_mask.view(batch_size, 1, 1, seg_len)
    return padding_mask


def attention(q, k, v, mask=None, dropout=None):
    scores = q.matmul(k.transpose(2,3))
    scores /= math.sqrt(q.shape[-1])

    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = torch.softmax(scores, dim=-1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

# compute positional embedding
class PostitionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = self.get_positional_model(d_model, seq_len)

    def get_positional_model(self, d_model, seq_len):
        pos_encoding = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        angle_rates = 1 / np.power(10000, (2 * (torch.arange(d_model).unsqueeze(0)//2)) / np.float32(d_model))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin((pos * angle_rates)[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos((pos * angle_rates)[:, 1::2])
        return pos_encoding

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.out_dim_per_head = d_model//n_heads

        self.query  = nn.Linear(self.d_model, self.n_heads * self.out_dim_per_head, bias= False)
        self.value  = nn.Linear(self.d_model, self.n_heads * self.out_dim_per_head, bias= False)
        self.key    = nn.Linear(self.d_model, self.n_heads * self.out_dim_per_head, bias= False)

        self.fc     = nn.Linear(self.d_model, self.n_heads * self.out_dim_per_head, bias= False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, v, k, q, mask=None):
        # input shape: (batch, sequence_size, embedding_size)
        # batch_size = v.shape[0]

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # (batch, sequence_size, n_head, emb_size_per_head)
        q, k, v = [self.split_heads(t) for t in (q,k,v)]

        # (batch, n_head, sequence_size, emb_size_per_head)
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]

        scores = attention(q, k, v, mask, self.dropout)

        # batch, sequence_size, emb_size
        scores = scores.transpose(1,2).contiguous()
        scores = scores.view(scores.shape[0], -1, self.d_model)

        # batch, sequence_size, emb_size
        out = self.fc(scores) 
        return out

 
class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.fc = nn.ReLU()
        self.linear2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        return self.linear2(self.fc(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, dff, d_model, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, dropout=0.1)
        self.fw  = FeedForward(d_model, dff)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        atten_out = self.mha(x, x, x, mask)
        atten_out = self.dropout(atten_out)
        output1 = self.layernorm(x + atten_out)

        output2 = self.fw(output1)
        output2 = self.dropout(output2)
        output2 = self.layernorm(output1 + output2)

        return output2


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, dff, d_model, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, dropout=0.1)
        self.fw  = FeedForward(d_model, dff)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, look_head_mask, padding_mask):
        atten_out = self.mha(x, x, x, look_head_mask)
        atten_out = self.dropout(atten_out)
        atten_out = self.layernorm(x + atten_out)

        atten_out2 = self.mha(v=encoder_output, k=encoder_output,
                              q = atten_out, mask = padding_mask)
        atten_out2 = self.dropout(atten_out2)
        atten_out2 = self.layernorm(atten_out + atten_out2)

        output = self.fw(atten_out2)
        output = self.dropout(output)
        output = self.layernorm(atten_out2 + output)

        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_layer, dff, max_len,
                       n_heads, vocab_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PostitionalEncoding(d_model, max_len)
        self.encoder_layers = [
            EncoderLayer(n_heads=n_heads, 
                         d_model=d_model, 
                         dff=dff, 
                         dropout=dropout)
            for _ in range(num_layer)
        ]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_layer, dff, max_len,
                       n_heads, vocab_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PostitionalEncoding(d_model, max_len)
        self.decoder_layers = [
            DecoderLayer(n_heads=n_heads, 
                    d_model=d_model, 
                    dff=dff, 
                    dropout=dropout)
            for _ in range(num_layer)
        ]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layer, dff, max_len,
                       input_vocab_size, target_vocab_size, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model=d_model,
                               dff=dff,
                               n_heads=n_heads,
                               max_len=max_len,
                               num_layer=num_layer,
                               vocab_size=input_vocab_size,
                               dropout=dropout)

        self.decoder = Decoder(d_model=d_model,
                               dff=dff,
                               n_heads=n_heads,
                               max_len=max_len,
                               num_layer=num_layer,
                               vocab_size=input_vocab_size,
                               dropout=dropout)

        self.linear = nn.Linear(d_model, target_vocab_size)
        self.fc = nn.Softmax()

    def create_masks(self, input, target):
        enc_padding_mask = create_padding_mask(input)
        dec_padding_mask = create_padding_mask(target)
        
        dec_in_look_head_mask = create_look_head_mask(target.shape[1])
        look_head_mask = torch.max(dec_padding_mask, dec_in_look_head_mask)
        return enc_padding_mask, look_head_mask

    def forward(self, inputs):
        input, target = inputs
        padding_mask, look_head_mask = self.create_masks(input, target)

        encoder_output = self.encoder(input, padding_mask)
        decoder_output = self.decoder(target, encoder_output, look_head_mask, padding_mask)

        final_layer = self.fc(self.linear(decoder_output))
        return final_layer
