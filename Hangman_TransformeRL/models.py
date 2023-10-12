import gym
import math
import numpy as np
import torch
from torch import nn

# DQN modified from BERT model implemented in https://wmathor.com/index.php/archives/1457/
# decoder added to this DQN

maxlen = 40
batch_size = 6
n_layers = 3
n_heads = 12
d_model = 256
d_ff = 256*4 # 4*d_model, FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
vocab_size = 29


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SeqEmbedding(nn.Module):
    def __init__(self):
        super(SeqEmbedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.dim() <= 1:
            x = x.view(1,-1)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)



class LetterEmbedding(nn.Module):
    def __init__(self):
        super(LetterEmbedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.dim() <= 1:
            x = x.view(1,-1)
        seq_len = x.size(1)
        embedding = self.tok_embed(x)
        return self.norm(embedding)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = Attention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual)  # output: [batch_size, seq_len, d_model]


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MHA()
        self.pos_ffn = FFN()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MHA()
        self.pos_ffn = FFN()

    def forward(self, dec_inputs, enc_outputs, enc_self_attn_mask):
        enc_self_attn_mask = enc_self_attn_mask[:, 0, :].unsqueeze(1)
        enc_self_attn_mask = enc_self_attn_mask.expand(-1, dec_inputs.shape[1], -1)
        dec_outputs = self.dec_self_attn(dec_inputs, enc_outputs, enc_outputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        dec_outputs = self.pos_ffn(dec_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return dec_outputs


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.fc2(self.softmax(self.fc1(x)))


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = SeqEmbedding()
        self.letter_embeddings = LetterEmbedding()
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.linear = FCN()

    def forward(self, enc_inputs, letters):
        words_emb = self.word_embeddings(enc_inputs)

        letters_emb = self.letter_embeddings(letters)
        if enc_inputs.dim() <= 1:
            enc_inputs = enc_inputs.view(1,-1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs)  # [batch_size, maxlen, maxlen]
        batch_size = words_emb.shape[0]
        letters_emb = letters_emb.expand(batch_size, -1, -1)

        for layer in self.encoder_layers:
            words_emb = layer(words_emb, enc_self_attn_mask)

        for layer in self.decoder_layers:
            letters_emb = layer(letters_emb, words_emb, enc_self_attn_mask)

        outputs = self.linear(letters_emb)

        return outputs
