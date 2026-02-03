import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 step_forward=1, past_input_size=0, forward_input_size=0,
                 dropout_rate=0.1, num_ids=None, id_embedding_size=16):
        """
        优化的LSTM模型，支持ID嵌入

        参数:
            input_size: 输入特征的总维度
            embedding_size: 输入值的嵌入层维度
            hidden_size: LSTM隐藏层的维度
            num_layers: LSTM的层数
            step_forward: 向前预测的步数，默认为1
            past_input_size: 过去输入特征的维度，默认为0
            forward_input_size: 向前预测的输入特征的维度，默认为0
            dropout_rate: dropout率，默认为0.1
            num_ids: ID的总数，如果为None则不使用ID嵌入
            id_embedding_size: ID嵌入的维度，默认为16
        """
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

        # 根据预测步数决定使用编码器-解码器结构还是单一LSTM
        if step_forward > 1:
            # 编码器
            encoder_input_size = input_size + embedding_size + (id_embedding_size if self.use_id_embedding else 0)
            self.encoder = nn.LSTM(encoder_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

            # 解码器
            decoder_input_size = (
                forward_input_size + embedding_size + (id_embedding_size if self.use_id_embedding else 0)
                if forward_input_size > 0 else embedding_size + (id_embedding_size if self.use_id_embedding else 0))
            self.decoder = nn.LSTM(decoder_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        else:
            # 单一LSTM
            lstm_input_size = input_size + embedding_size + (id_embedding_size if self.use_id_embedding else 0)
            self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # 输出层
        self.out_net = nn.Linear(hidden_size, 1)

    def forward(self, X=None, y=None, X_forward=None, ids=None, is_training=False):
        """
        前向传播

        参数:
            X: 输入特征张量 (batch, seq_len, input_size)
            y: 目标值张量 (batch, seq_len + step_forward - 1, 1)
            X_forward: 向前预测的输入特征 (batch, seq_len + step_forward, forward_input_size)
            ids: ID张量 (batch,_)
            is_training: 是否在训练模式

        返回:
            (None, predictions) 元组
        """
        # 初始化隐藏状态
        device = X.device if X is not None else X_forward.device
        batch_size = y.size(0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

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
            y_pred = self._multi_step_prediction(X_concat, y, X_forward, h, c, batch_size, seq_len, is_training, ids)
            return None, y_pred
        else:
            # 单步预测模式
            y_input = y[:, :seq_len, :]
            y_embed = self.input_embed(y_input)
            x = torch.cat([X_concat, y_embed], dim=2)
            out, _ = self.lstm(x, (h, c))

            out = self.out_net(out[:, -1:, :])
            return None, out

    def _multi_step_prediction(self, X_concat, y, X_forward, h, c, batch_size, seq_len, is_training, ids=None):
        """处理多步预测逻辑"""
        y_pred = []

        # 第一步预测
        y_input = y[:, :seq_len, :]
        y_embed = self.input_embed(y_input)
        x = torch.cat([X_concat, y_embed], dim=2)
        out, (h, c) = self.encoder(x, (h, c))

        y_next = self.out_net(out[:, -1, :]).view(batch_size, -1)
        y_pred.append(y_next)

        if is_training:
            # 训练模式 - 使用真实值作为输入
            if self.forward_input_size > 0:
                y_input = y[:, seq_len:seq_len + self.step_forward - 1, :]
                y_embed = self.input_embed(y_input)
                x = X_forward[:, seq_len + 1:seq_len + self.step_forward, :]
                # 处理ID嵌入
                if self.use_id_embedding and ids is not None:
                    id_emb = self.id_embedding(ids)  # (batch, id_embedding_size)
                    id_emb = id_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, step_forward-1, id_embedding_size)
                    x = torch.cat([x, id_emb, y_embed], dim=2)
                else:
                    x = torch.cat([x, y_embed], dim=2)
                out, _ = self.decoder(x, (h, c))

                out = self.out_net(out).view(batch_size, -1)
                y_pred = torch.cat([y_pred[0], out], dim=1).unsqueeze(2)
            else:
                y_input = y[:, seq_len:seq_len + self.step_forward - 1, :]
                y_embed = self.input_embed(y_input)
                # 处理ID嵌入
                if self.use_id_embedding and ids is not None:
                    id_emb = self.id_embedding(ids)  # (batch, id_embedding_size)
                    id_emb = id_emb.unsqueeze(1).expand(-1, y_embed.size(1), -1)
                    x = torch.cat([id_emb, y_embed], dim=2)
                else:
                    x = y_embed
                out, _ = self.decoder(x, (h, c))

                out = self.out_net(out).view(batch_size, -1)
                y_pred = torch.cat([y_pred[0], out], dim=1).unsqueeze(2)
        else:
            # 预测模式 - 使用预测值作为输入
            for s in range(seq_len, seq_len + self.step_forward - 1):
                if self.forward_input_size > 0:
                    y_embed = self.input_embed(y_next).view(batch_size, -1)
                    x = X_forward[:, s + 1, :].view(batch_size, -1)
                    # 处理ID嵌入
                    if self.use_id_embedding and ids is not None:
                        id_emb = self.id_embedding(ids)  # (batch, id_embedding_size)
                        x = torch.cat([x, id_emb, y_embed], dim=1)
                    else:
                        x = torch.cat([x, y_embed], dim=1)
                    inp = x.unsqueeze(1)
                    out, (h, c) = self.decoder(inp, (h, c))

                    y_next = self.out_net(out).view(batch_size, -1)
                else:
                    y_embed = self.input_embed(y_next).view(batch_size, -1)
                    # 处理ID嵌入
                    if self.use_id_embedding and ids is not None:
                        id_emb = self.id_embedding(ids)  # (batch, id_embedding_size)
                        inp = torch.cat([id_emb.unsqueeze(1), y_embed.unsqueeze(1)], dim=2)
                    else:
                        inp = y_embed.unsqueeze(1)
                    out, (h, c) = self.decoder(inp, (h, c))

                    y_next = self.out_net(out).view(batch_size, -1)

                y_pred.append(y_next)

            y_pred = torch.cat(y_pred, dim=1).unsqueeze(2)

        return y_pred