import torch
from torch import nn as nn

from models.base import LSTMAttentionEncoder, LSTMAttentionDecoder, RevIN

class FeatureTower(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, step_forward=1, past_input_size=0, forward_input_size=0,
                 attn_head=2, share_outNet=True, dropout_rate=0.1):
        super(FeatureTower, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet

        # 编码器
        self.encoder = LSTMAttentionEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            att_head=attn_head,
            dropout_rate=dropout_rate
        )

        # 输出层
        self.outNet = nn.Linear(hidden_size, 1)

        # 解码器
        if step_forward > 1 and forward_input_size > 0:
            self.decoder = LSTMAttentionDecoder(
                input_size=forward_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                att_head=attn_head,
                dropout_rate=dropout_rate
            )
            if not share_outNet:
                self.decoder_outNet = nn.Linear(hidden_size, 1)

    def _process_feat(self, X, X_forward):
        # 处理输入特征
        if self.forward_input_size > 0:
            if self.past_input_size > 0:
                seq_len = X.size(1)
                X_concat = torch.cat((X, X_forward[:, 1: seq_len + 1, :]), dim=2)
            else:
                seq_len = X_forward.size(1) - self.step_forward
                X_concat = X_forward[:, 1: seq_len + 1, :]
        else:
            seq_len = X.size(1)
            X_concat = X
        return X_concat, seq_len

    def forward(self, X=None, X_forward=None, is_training=False, ids=None):
        X_concat, seq_len = self._process_feat(X, X_forward)

        # 编码器
        outs, h, c, K, V = self.encoder(X_concat)
        preds = self.outNet(outs)

        # 单步预测 或者 没有未来特征
        if self.step_forward == 1 or self.forward_input_size <= 0:
            return outs, preds

        # 多步预测 且 有未来特征
        encoder_k_and_v = {'k': K, 'v': V}
        if is_training:
            x = X_forward[:, seq_len + 1: seq_len + self.step_forward, :]
            dec_out, h, c, K, V = self.decoder(x, h, c, encoder_k_and_v)
            if self.share_outNet:
                dec_pred = self.outNet(dec_out)
            else:
                dec_pred = self.decoder_outNet(dec_out)

            # 组装结果
            outs = torch.cat([outs, dec_out], dim=1)
            preds = torch.cat([preds, dec_pred], dim=1)
        else:
            for s in range(seq_len, seq_len + self.step_forward - 1):
                x = X_forward[:, s + 1: s + 2, :]
                dec_out, h, c, K_, V_ = self.decoder(x, h, c, encoder_k_and_v)

                if self.share_outNet:
                    dec_pred = self.outNet(dec_out)
                else:
                    dec_pred = self.decoder_outNet(dec_out)

                encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)

                outs = torch.cat([outs, dec_out], dim=1)
                preds = torch.cat([preds, dec_pred], dim=1)

        return outs, preds


class TargetTower(nn.Module):
    def __init__(self, embedding_size, hidden_size_feat, hidden_size_target, num_layers, step_forward=1,
                 past_input_size=0, forward_input_size=0, attn_head=2, share_outNet=True):
        super(TargetTower, self).__init__()

        self.hidden_size_feat = hidden_size_feat
        self.hidden_size_target = hidden_size_target
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet and forward_input_size > 0

        # 初始化网络结构
        self.input_embed = nn.Linear(1, embedding_size)
        # 编码器
        self.encoder = LSTMAttentionEncoder(
            input_size=embedding_size,
            hidden_size=hidden_size_target,
            num_layers=num_layers,
            att_head=attn_head
        )
        self.outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)

        # 解码器
        if step_forward > 1:
            self.decoder = LSTMAttentionDecoder(
                input_size=embedding_size,
                hidden_size=hidden_size_target,
                num_layers=num_layers,
                att_head=attn_head
            )

            # 定义为非share_outNet 和 多步预测且没有未来输入特征时 增加一个decoder_out_Net
            if not self.share_outNet:
                if forward_input_size > 0:
                    self.decoder_outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)
                else:
                    self.decoder_outNet = nn.Linear(hidden_size_target, 1)


    def forward(self, y, out_x, is_training=False):
        _, seq_total, _ = y.size()
        seq_len = seq_total - self.step_forward

        y_input = y[:, : seq_len, :]
        yembed = self.input_embed(y_input)

        out_y, h, c, K, V = self.encoder(yembed)
        out_x_past = out_x[:, :seq_len, :]
        out_y = torch.cat([out_x_past, out_y], dim=2)  # num_ts, num_features + embedding
        pred_y = self.outNet(out_y[:, -1:, :])

        # 根据预测步骤组织输出
        if self.step_forward == 1:
            return pred_y

        # step_forward > 1
        encoder_k_and_v = {'k': K, 'v': V}
        if is_training:
            y_input = y[:, seq_len: seq_len + self.step_forward - 1, :]
            yembed = self.input_embed(y_input)
            out_y_d, h, c, K, V = self.decoder(yembed, h, c, encoder_k_and_v)

            if self.forward_input_size > 0:
                # 存在未来特征
                out_x_d = out_x[:, seq_len: seq_len + self.step_forward - 1, :]
                out_y_d = torch.cat([out_x_d, out_y_d], dim=2)  # num_ts, num_features + embedding
            if self.share_outNet:
                pred_y_d = self.outNet(out_y_d)
            else:
                pred_y_d = self.decoder_outNet(out_y_d)

            pred_y = torch.cat([pred_y, pred_y_d], dim=1)
        else:
            ynext = pred_y
            for s in range(seq_len, seq_len + self.step_forward - 1):
                yembed = self.input_embed(ynext)
                out_y_d, h, c, K_, V_ = self.decoder(yembed, h, c, encoder_k_and_v)
                if self.forward_input_size > 0:
                    out_x_d = out_x[:, s: s + 1, :]
                    out_y_d = torch.cat([out_x_d, out_y_d], dim=2)

                if self.share_outNet:
                    pred_y_d = self.outNet(out_y_d)
                else:
                    pred_y_d = self.decoder_outNet(out_y_d)

                ynext = pred_y_d
                encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)
                # 组装结果
                pred_y = torch.cat([pred_y, pred_y_d], dim=1)

        return pred_y

#  定义 LSTM + PostMaskAttention + causal + Adversarial 类
class LSTMPostMarkAttCausalAd(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size_feat, hidden_size_target, num_layers,
                step_forward=1, past_input_size=0, forward_input_size=0, attn_head_feat=2,
                 attn_head_target=2, share_outNet=True, dropout_rate=0.1):

        super(LSTMPostMarkAttCausalAd, self).__init__()
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward

        self.feature_tower = FeatureTower(
            input_size=input_size,
            hidden_size=hidden_size_feat,
            num_layers=num_layers,
            step_forward=step_forward,
            past_input_size=past_input_size,
            forward_input_size=forward_input_size,
            attn_head=attn_head_feat,
            share_outNet=share_outNet,
            dropout_rate=dropout_rate
        )

        self.target_tower = TargetTower(
            embedding_size=embedding_size,
            hidden_size_feat=hidden_size_feat,
            hidden_size_target=hidden_size_target,
            num_layers=num_layers,
            step_forward=step_forward,
            past_input_size=past_input_size,
            forward_input_size=forward_input_size,
            attn_head=attn_head_target,
            share_outNet=share_outNet,
            dropout_rate=dropout_rate
        )

    def forward(self, X=None, y=None, X_forward=None, is_training=False):
        out_x, pred_x = self.feature_tower(X, X_forward, is_training)
        pred_y = self.target_tower(y, out_x, is_training)
        return pred_x, pred_y
'''    
#  定义 LSTM + PostMaskAttention + causal + Adversarial 类
class LSTMPostMarkAttCausalAd(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size_feat, hidden_size_target, num_layers,
                step_forward=1, past_input_size=0, forward_input_size=0, attn_head_feat=2,
                 attn_head_target=2, share_outNet=True, dropout_rate=0.1):

        super(LSTMPostMarkAttCausalAd, self).__init__()
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward

        # 1. 初始化 RevIN
        # num_features=1 因为 Target 通常是单变量 (batch, seq, 1)
        self.revin = RevIN(num_features=1, affine=True)

        self.feature_tower = FeatureTower(
            input_size=input_size,
            hidden_size=hidden_size_feat,
            num_layers=num_layers,
            step_forward=step_forward,
            past_input_size=past_input_size,
            forward_input_size=forward_input_size,
            attn_head=attn_head_feat,
            share_outNet=share_outNet,
            dropout_rate=dropout_rate
        )

        self.target_tower = TargetTower(
            embedding_size=embedding_size,
            hidden_size_feat=hidden_size_feat,
            hidden_size_target=hidden_size_target,
            num_layers=num_layers,
            step_forward=step_forward,
            past_input_size=past_input_size,
            forward_input_size=forward_input_size,
            attn_head=attn_head_target,
            share_outNet=share_outNet
            # dropout_rate=dropout_rate
        )

    def forward(self, X=None, y=None, X_forward=None, is_training=False, ids=None):
        # out_x, pred_x = self.feature_tower(X, X_forward, is_training)
        # pred_y = self.target_tower(y, out_x, is_training)

        # 2. RevIN 预处理：Instance Normalization
        # 关键点：只能使用历史数据 (Past) 来计算均值和标准差，防止未来信息泄露
        seq_len_past = y.size(1) - self.step_forward

        # 提取历史部分用于计算统计量
        y_past = y[:, :seq_len_past, :]

        # 计算统计量 (mean, std) 并存储在 revin 实例中
        # 注意：这里调用 _get_statistics 仅更新统计量，不返回
        self.revin._get_statistics(y_past)

        # 使用刚刚计算的历史统计量，归一化整个序列 y (包含训练所需的未来部分)
        # 这样模型在 Teacher Forcing 时看到的未来值也是基于历史分布归一化的
        y_norm = self.revin._normalize(y)

        # 3. 模型前向传播 (使用归一化后的 y_norm)
        out_x, pred_x = self.feature_tower(X, X_forward, ids, is_training)
        pred_y_norm = self.target_tower(y_norm, out_x, is_training)

        # 4. RevIN 后处理：Denormalization
        # 将模型输出的归一化预测值还原回原始数据尺度
        # 这样外部的 Loss 计算和 Metrics 评估可以使用真实的物理数值
        pred_y = self.revin._denormalize(pred_y_norm)

        return pred_x, pred_y
'''