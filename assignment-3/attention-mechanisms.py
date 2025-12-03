import torch
import torch.nn as nn
import torch.nn.functional as F

#scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shapes: (batch, heads, seq_len, head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # dot product
        scores = scores / (Q.size(-1) ** 0.5)          # scale factor

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

#self attention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attn = ScaledDotProductAttention()

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        output, weights = self.attn(Q, K, V)
        return output, weights

#masked self attention
class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attn = ScaledDotProductAttention()

    def forward(self, x):
        seq_len = x.size(1)

        # Lower triangular masking
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        output, weights = self.attn(Q, K, V, mask=mask)
        return output, weights
#multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention()
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn = self.attention(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out(out)
        return out, attn

#cross attention
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention()

    def forward(self, decoder_hidden, encoder_output):
        Q = self.W_q(decoder_hidden)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)

        out, attn = self.attention(Q, K, V)
        return out, attn

x = torch.randn(1, 5, 64)   # (batch, seq_len, embed_dim)
enc = torch.randn(1, 5, 64)

print("Self Attention:")
sa = SelfAttention(64)
out, _ = sa(x)
print(out.shape)

print("\nMasked Self Attention:")
msa = MaskedSelfAttention(64)
out, _ = msa(x)
print(out.shape)

print("\nMulti-Head Attention:")
mha = MultiHeadAttention(64, num_heads=8)
out, _ = mha(x)
print(out.shape)

print("\nCross Attention:")
ca = CrossAttention(64)
out, _ = ca(x, enc)
print(out.shape)
