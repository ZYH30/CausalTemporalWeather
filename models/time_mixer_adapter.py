import torch
import torch.nn as nn
from types import SimpleNamespace
from models import TimeMixer_native 

class TimeMixerAdapter(nn.Module):
    def __init__(self, args):
        super(TimeMixerAdapter, self).__init__()
        
        self.configs = SimpleNamespace()
        self.configs.task_name = 'long_term_forecast'
        self.configs.is_training = 1 
        self.configs.use_gpu = torch.cuda.is_available()
        self.configs.device_ids = [0]
        
        # === 维度配置 ===
        self.configs.enc_in = args.past_input_size    
        self.configs.dec_in = args.past_input_size    
        
        # [关键 1] 顺应模型架构: 必须预测所有通道
        # 只有这样，Linear层输出的 [B, T, 11] 才能和 趋势项 [B, T, 11] 正确相加
        self.configs.c_out = args.past_input_size     
        
        # [关键 2] 使用混合模式
        self.configs.channel_independence = args.channel_independence 
        
        # 动态参数读取
        self.configs.d_model = args.embedding_size if hasattr(args, 'embedding_size') else 16
        self.configs.d_ff = args.d_ff if hasattr(args, 'd_ff') else self.configs.d_model * 4
        self.configs.dropout = args.dropout_rate if hasattr(args, 'dropout_rate') else 0.1
        self.configs.embed = 'timeF' 
        self.configs.freq = 'h'      
        
        self.configs.seq_len = args.sequence_length
        self.configs.label_len = int(args.sequence_length / 2)
        self.configs.pred_len = args.step_forward
        
        self.configs.e_layers = args.num_layers 
        self.configs.down_sampling_layers = args.down_sampling_layers if hasattr(args, 'down_sampling_layers') else 3
        self.configs.down_sampling_window = args.down_sampling_window if hasattr(args, 'down_sampling_window') else 2
        self.configs.down_sampling_method = 'avg'
        
        self.configs.use_future_temporal_feature = 0 
        self.configs.use_norm = 1 
        self.configs.decomp_method = 'moving_avg'
        self.configs.moving_avg = 25 
        
        self.model = TimeMixer_native.Model(self.configs)

    def forward(self, X, y=None, X_forward=None, ids=None, is_training=False, x_mark=None):
        x_enc = X
        
        # 时间特征处理
        if x_mark is not None:
            x_mark_enc = x_mark[:, :self.configs.seq_len, :]
            dec_start_idx = self.configs.seq_len - self.configs.label_len
            x_mark_dec = x_mark[:, dec_start_idx:, :]
        else:
            x_mark_enc = None
            x_mark_dec = None
        
        # Decoder 输入
        B, _, D = x_enc.shape
        dec_inp_token = x_enc[:, -self.configs.label_len:, :]
        dec_inp_zero = torch.zeros([B, self.configs.pred_len, D], device=x_enc.device, dtype=x_enc.dtype)
        dec_inp = torch.cat([dec_inp_token, dec_inp_zero], dim=1)
        
        # === 前向传播 ===
        # 输出维度: [B, Pred, enc_in] (即 11 维)
        # 此时: 第 i 个通道 = 神经网络预测的第 i 个特征增量 + 第 i 个特征的历史趋势
        # 物理意义完全正确。
        dec_out = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
        
        # === 输出对齐 ===
        # 1. 截取预测时间步
        dec_out = dec_out[:, -self.configs.pred_len:, :]
        
        # 2. [关键 3] 精准提取 Target
        # 因为我们在 util.py 中强制将 Target 放在了最后一位
        # 所以这里的 -1 通道绝对就是 Target 的预测值
        dec_out = dec_out[:, :, -1:]
        
        return None, dec_out