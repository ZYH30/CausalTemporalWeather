import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 step_forward=1, past_input_size=0, forward_input_size=0,
                 dropout_rate=0.1, num_ids=None, id_embedding_size=16,
                 nhead=4, dim_feedforward=512):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.step_forward = step_forward
        self.past_input_size = past_input_size
        self.forward_input_size = forward_input_size
        self.use_id_embedding = num_ids is not None

        # ID嵌入层
        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(num_ids, id_embedding_size)
        else:
            self.id_embedding = None

        # 输入值嵌入层
        self.input_embed = nn.Linear(1, embedding_size)

        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_size, dropout_rate)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器部分（用于多步预测）
        if step_forward > 1:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 编码器输入投影层
        self.encoder_input_proj = nn.Linear(
            input_size + embedding_size + (id_embedding_size if self.use_id_embedding else 0),
            hidden_size
        )

        # 解码器输入投影层（处理不同输入情况）
        if forward_input_size > 0:
            self.decoder_input_proj = nn.Linear(
                forward_input_size + embedding_size + (id_embedding_size if self.use_id_embedding else 0),
                hidden_size
            )
        else:
            self.decoder_input_proj = nn.Linear(
                embedding_size + (id_embedding_size if self.use_id_embedding else 0),
                hidden_size
            )

        # 输出层
        self.out_net = nn.Linear(hidden_size, 1)

    def forward(self, X=None, y=None, X_forward=None, ids=None, is_training=False):
        device = X.device if X is not None else X_forward.device
        batch_size = y.size(0)

        # 处理ID嵌入
        if self.use_id_embedding and ids is not None:
            id_emb = self.id_embedding(ids)  # (batch, id_embedding_size)
            # 扩展ID嵌入以匹配序列长度
            id_emb = id_emb.unsqueeze(1).expand(-1, X.size(1), -1)  # (batch, seq_len, id_embedding_size)

        # 处理输入特征
        if self.forward_input_size > 0:
            if self.past_input_size > 0:
                batch_size, seq_len, _ = X.size()
                X_concat = torch.cat((X, X_forward[:, 1:seq_len + 1, :]), dim=2)
            else:
                batch_size, seq_len, _ = X_forward.size()
                seq_len = seq_len - self.step_forward
                X_concat = X_forward[:, 1:seq_len + 1, :]
        else:
            batch_size, seq_len, _ = X.size()
            X_concat = X

        # 拼接ID嵌入
        if self.use_id_embedding and ids is not None:
            X_concat = torch.cat([X_concat, id_emb], dim=2)

        if self.step_forward > 1:
            # 多步预测模式
            y_pred = self._multi_step_prediction(X_concat, y, X_forward, batch_size, seq_len, is_training, ids)
            return None, y_pred
        else:
            # 单步预测模式
            y_input = y[:, :seq_len, :]
            y_embed = self.input_embed(y_input)
            x = torch.cat([X_concat, y_embed], dim=2)

            # 投影到隐藏维度
            x = self.encoder_input_proj(x)

            # 添加位置编码
            x = self.positional_encoding(x)

            # Transformer编码
            memory = self.transformer_encoder(x)

            # 预测
            out = self.out_net(memory[:, -1:, :])
            return None, out

    def _multi_step_prediction(self, X_concat, y, X_forward, batch_size, seq_len, is_training, ids=None):
        """处理多步预测逻辑"""
        y_pred = []

        # 第一步预测
        y_input = y[:, :seq_len, :]
        y_embed = self.input_embed(y_input)
        x = torch.cat([X_concat, y_embed], dim=2)

        # 投影到隐藏维度
        x = self.encoder_input_proj(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        memory = self.transformer_encoder(x)

        # 初始预测
        y_next = self.out_net(memory[:, -1:, :]).view(batch_size, -1)
        y_pred.append(y_next)

        if is_training:
            # 训练模式 - 使用真实值作为输入
            if self.forward_input_size > 0:
                y_input = y[:, seq_len:seq_len + self.step_forward - 1, :]
                y_embed = self.input_embed(y_input)
                x = X_forward[:, seq_len + 1:seq_len + self.step_forward, :]

                # 处理ID嵌入
                if self.use_id_embedding and ids is not None:
                    id_emb = self.id_embedding(ids)
                    id_emb = id_emb.unsqueeze(1).expand(-1, x.size(1), -1)
                    x = torch.cat([x, id_emb, y_embed], dim=2)
                else:
                    x = torch.cat([x, y_embed], dim=2)

                # 使用解码器投影层
                tgt = self.decoder_input_proj(x)

                # 添加位置编码
                tgt = self.positional_encoding(tgt)

                # Transformer解码
                out = self.transformer_decoder(tgt, memory)

                out = self.out_net(out).view(batch_size, -1)
                y_pred = torch.cat([y_pred[0], out], dim=1).unsqueeze(2)
            else:
                y_input = y[:, seq_len:seq_len + self.step_forward - 1, :]
                y_embed = self.input_embed(y_input)

                # 处理ID嵌入
                if self.use_id_embedding and ids is not None:
                    id_emb = self.id_embedding(ids)
                    id_emb = id_emb.unsqueeze(1).expand(-1, y_embed.size(1), -1)
                    x = torch.cat([id_emb, y_embed], dim=2)
                else:
                    x = y_embed

                # 使用解码器投影层
                tgt = self.decoder_input_proj(x)

                # 添加位置编码
                tgt = self.positional_encoding(tgt)

                # Transformer解码
                out = self.transformer_decoder(tgt, memory)

                out = self.out_net(out).view(batch_size, -1)
                y_pred = torch.cat([y_pred[0], out], dim=1).unsqueeze(2)
        else:
            # 预测模式 - 使用预测值作为输入
            tgt_all = []
            for s in range(seq_len, seq_len + self.step_forward - 1):
                if self.forward_input_size > 0:
                    y_embed = self.input_embed(y_next).view(batch_size, -1)
                    x = X_forward[:, s + 1, :].view(batch_size, -1)

                    # 处理ID嵌入
                    if self.use_id_embedding and ids is not None:
                        id_emb = self.id_embedding(ids)
                        x = torch.cat([x, id_emb, y_embed], dim=1)
                    else:
                        x = torch.cat([x, y_embed], dim=1)

                    # 使用解码器投影层
                    tgt = self.decoder_input_proj(x.unsqueeze(1))

                    # 添加位置编码
                    tgt = self.positional_encoding(tgt)

                    tgt_all.append(tgt)
                else:
                    y_embed = self.input_embed(y_next).view(batch_size, -1)

                    # 处理ID嵌入
                    if self.use_id_embedding and ids is not None:
                        id_emb = self.id_embedding(ids)
                        tgt = torch.cat([id_emb.unsqueeze(1), y_embed.unsqueeze(1)], dim=2)
                    else:
                        tgt = y_embed.unsqueeze(1)

                    # 使用解码器投影层
                    tgt = self.decoder_input_proj(tgt)

                    # 添加位置编码
                    tgt = self.positional_encoding(tgt)

                    tgt_all.append(tgt)

            # 合并所有目标序列
            tgt_seq = torch.cat(tgt_all, dim=1)

            # Transformer解码
            out = self.transformer_decoder(tgt_seq, memory)

            # 预测
            y_next = self.out_net(out).view(batch_size, -1)
            y_pred.append(y_next)

            y_pred = torch.cat(y_pred, dim=1).unsqueeze(2)

        return y_pred


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)