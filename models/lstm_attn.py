import torch
from torch import nn as nn

from models.base import MaskedSelfAttention


#  定义 LSTM + PostMaskAttention 类
class LSTMPostMarkAtt(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, step_forward=1, past_input_size=0,
                 forward_input_size=0, att_head=2, dropout_rate=0.1):
        """
        input_size: 输入特征的维度(包括可变特征和不变特征，总的输入维度)
        embedding_size: 嵌入层的维度
        hidden_size: LSTM隐藏层的维度
        num_layers: LSTM的层数
        step_forward: 向前预测的步数，默认为1
        forward_feature: 是否使用向前预测的特征，默认为True
        forward_feature_add: 是否将向前预测的特征与输入特征相加，默认为True
        forward_input_size: 向前预测的输入特征的维度，默认为3
        """

        super(LSTMPostMarkAtt, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.step_forward = step_forward
        self.past_input_size = past_input_size
        self.forward_input_size = forward_input_size

        self.input_embed = nn.Linear(1, embedding_size)
        if step_forward > 1:
            self.encoder = nn.LSTM(input_size + embedding_size, hidden_size, num_layers, batch_first=True,
                                   dropout=dropout_rate)
            self.encoderAttn = MaskedSelfAttention(hidden_size, att_head)
            self.decoderAttn = MaskedSelfAttention(hidden_size, att_head)
            if forward_input_size > 0:
                self.decoder = nn.LSTM(forward_input_size + embedding_size, hidden_size, num_layers, batch_first=True,
                                       dropout=dropout_rate)
            else:
                self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        else:
            self.lstm = nn.LSTM(input_size + embedding_size, hidden_size, num_layers, batch_first=True,
                                dropout=dropout_rate)
            self.Attn = MaskedSelfAttention(hidden_size, att_head)
            # input tensor: (batch_size, seq_len, input_size) output tensor: (batch_size, seq_len, input_size)

        self.outNet = nn.Linear(hidden_size, 1)

    def forward(self, X=None, y=None, X_forward=None, is_training=False, ids=None):
        if self.forward_input_size > 0:
            if self.past_input_size > 0:
                bat, seq_len, X_dim = X.size()
                X_concat = torch.cat((X, X_forward[:, 1: seq_len + 1, :]), dim=2)
            else:
                bat, seq_len, X_forward_dim = X_forward.size()
                seq_len = seq_len - self.step_forward
                X_concat = X_forward[:, 1:seq_len + 1, :]
        else:
            bat, seq_len, X_dim = X.size()
            X_concat = X

        h = torch.zeros(self.num_layers, y.size(0), self.hidden_size, device=X_concat.device)
        c = torch.zeros(self.num_layers, y.size(0), self.hidden_size, device=X_concat.device)

        if self.step_forward > 1:
            ypred = []

            y_input = y[:, : seq_len, :]
            yembed = self.input_embed(y_input)
            x = torch.cat([X_concat, yembed], dim=2)  # num_ts, num_features + embedding
            out, (h, c) = self.encoder(x, (h, c))  # LSTM会自动处理步特征
            out, _, K, V = self.encoderAttn(out, mask=True)
            ynext = self.outNet(out[:, -1, :]).view(bat, -1)
            ypred.append(ynext)
            encoder_k_and_v = {'k': K, 'v': V}

            if is_training:
                if self.forward_input_size > 0:
                    y_input = y[:, seq_len: seq_len + self.step_forward - 1, :]
                    yembed = self.input_embed(y_input)

                    x = X_forward[:, seq_len + 1: seq_len + self.step_forward, :]
                    x = torch.cat([x, yembed], dim=2)  # num_ts, num_features + embedding

                    out, (_, _) = self.decoder(x, (h, c))
                    out, _, K_, V_ = self.decoderAttn(out, mask=True, forward_k_and_v=encoder_k_and_v)

                    out = self.outNet(out).view(bat, -1)
                    ypred = torch.cat([ypred[0], out], dim=1).unsqueeze(2)
                else:
                    y_input = y[:, seq_len: seq_len + self.step_forward - 1, :]
                    yembed = self.input_embed(y_input)

                    out, (_, _) = self.decoder(yembed, (h, c))
                    out, _, K_, V_ = self.decoderAttn(out, mask=True, forward_k_and_v=encoder_k_and_v)

                    # lstm的输出 h, c 会传递到解码器
                    out = self.outNet(out).view(bat, -1)
                    ypred = torch.cat([ypred[0], out], dim=1).unsqueeze(2)

                return None, ypred
            else:
                for s in range(seq_len, seq_len + self.step_forward - 1):
                    if self.forward_input_size > 0:
                        yembed = self.input_embed(ynext).view(bat, -1)
                        x = X_forward[:, s + 1, :].view(bat, -1)

                        x = torch.cat([x, yembed], dim=1)  # num_ts, num_features + embedding
                        inp = x.unsqueeze(1)
                        out, (h, c) = self.decoder(inp, (h, c))

                        out, _, K_, V_ = self.decoderAttn(out, mask=False, forward_k_and_v=encoder_k_and_v)
                        encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                        encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)

                        ynext = self.outNet(out).view(bat, -1)
                    else:
                        yembed = self.input_embed(ynext).view(bat, -1)
                        inp = yembed.unsqueeze(1)
                        out, (h, c) = self.decoder(inp, (h, c))
                        out, _, K_, V_ = self.decoderAttn(out, mask=False, forward_k_and_v=encoder_k_and_v)
                        encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                        encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)

                        ynext = self.outNet(out).view(bat, -1)

                    ypred.append(ynext)

                # 循环结束后，ypred 中包含了所有的预测值，将其拼接并返回
                ypred = torch.cat(ypred, dim=1).unsqueeze(2)
                return None, ypred
        else:
            y_input = y[:, : seq_len, :]
            yembed = self.input_embed(y_input)
            x = torch.cat([X_concat, yembed], dim=2)  # num_ts, num_features + embedding
            out, _ = self.lstm(x, (h, c))  # LSTM会自动处理步特征
            out, _, K, V = self.Attn(out, mask=True)

            out = self.outNet(out[:, -1:, :])
            return None, out