import torch
import torch.nn.functional as F
from torch import nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size_out, heads, embed_size_in=None):
        super().__init__()
        self.embed_size_out = embed_size_out
        self.heads = heads
        self.head_dim = embed_size_out // heads

        assert self.head_dim * heads == embed_size_out, "嵌入维度必须是头数的整数倍"

        if embed_size_in is None:
            embed_size_in = embed_size_out
        self.values = nn.Linear(embed_size_in, embed_size_out)
        self.keys = nn.Linear(embed_size_in, embed_size_out)
        self.queries = nn.Linear(embed_size_in, embed_size_out)
        # self.fc_out = nn.Linear(embed_size_in, embed_size_out)

    def forward(self, x, mask=True, forward_k_and_v=None):
        batch_size, seq_length, _ = x.size()

        # 参数验证
        if forward_k_and_v is not None:
            if not isinstance(forward_k_and_v, dict) or 'k' not in forward_k_and_v or 'v' not in forward_k_and_v:
                raise ValueError("forward_k_and_v must be a dict containing 'k' and 'v' tensors")

            k_prev, v_prev = forward_k_and_v['k'], forward_k_and_v['v']
            if (k_prev.size(0) != batch_size or v_prev.size(0) != batch_size or
                    k_prev.size(2) != self.embed_size_out or v_prev.size(2) != self.embed_size_out):
                raise ValueError('batch_size and embed_size must match between x and forward_k_and_v')

            if k_prev.device != x.device or v_prev.device != x.device:
                raise ValueError('forward_k_and_v tensors must be on the same device as x')

        # 线性变换
        V_current = self.values(x)  # (batch_size, seq_len, embed_size)
        K_current = self.keys(x)  # (batch_size, seq_len, embed_size)
        Q = self.queries(x)  # (batch_size, seq_len, embed_size)

        # 处理前向传递的k和v
        if forward_k_and_v is not None:
            K = torch.cat([forward_k_and_v['k'], K_current], dim=1)  # (batch_size, seq_prev + seq_len, embed_size)
            V = torch.cat([forward_k_and_v['v'], V_current], dim=1)  # (batch_size, seq_prev + seq_len, embed_size)
            seq_total = K.size(1)
        else:
            K, V = K_current, V_current
            seq_total = seq_length

        # 分割多头
        V_split = V.view(batch_size, seq_total, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K_split = K.view(batch_size, seq_total, self.heads, self.head_dim).permute(0, 2, 1, 3)
        Q_split = Q.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数 (QK^T)
        attn_scores = torch.matmul(Q_split, K_split.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # 应用注意力掩码
        if mask is not None:
            if forward_k_and_v is not None:
                # 创建因果掩码，只允许关注当前位置及之前的token
                seq_prev = forward_k_and_v['k'].size(1)
                causal_mask = torch.ones(seq_length, seq_total, dtype=torch.bool, device=x.device)
                for i in range(seq_length):
                    causal_mask[i, :seq_prev + i + 1] = False  # False表示需要保留的位置
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            else:
                # 标准的下三角掩码
                causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        # 计算注意力权重
        attention = F.softmax(attn_scores, dim=-1)

        # 加权求和
        out = torch.matmul(attention, V_split)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_length, self.embed_size_out)
        # out = self.fc_out(out)

        return out, attention, K_current, V_current


class LSTMAttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, att_head, dropout_rate=0.1):
        super(LSTMAttentionEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attn = MaskedSelfAttention(hidden_size, att_head)

    def forward(self, x):
        batch_size, _, _ = x.size()
        device = x.device

        # Initialize hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Process inputs
        out, (h, c) = self.lstm(x, (h, c))

        # Apply attention
        out, _, K, V = self.attn(out, mask=True)

        return out, h, c, K, V


class LSTMAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, att_head, dropout_rate=0.1):
        super(LSTMAttentionDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attn = MaskedSelfAttention(hidden_size, att_head)

    def forward(self, x, h, c, encoder_k_v):
        out_d, (h, c) = self.lstm(x, (h, c))
        out_d, _, K, V = self.attn(out_d, forward_k_and_v=encoder_k_v)

        return out_d, h, c, K, V


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        batch_size, _, _ = x.size()
        device = x.device

        # Initialize hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Process inputs
        out, (h, c) = self.encoder(x, (h, c))

        return out, h, c


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(LSTMDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, x, h, c):
        out, (h, c) = self.decoder(x, (h, c))

        return out, h, c