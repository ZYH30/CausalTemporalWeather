import argparse
from html import parser

import yaml


def create_parser():
    parser = argparse.ArgumentParser(description='时间序列预测模型参数')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/weather_sourceData.csv', # returnForcast.parquet
                        help='训练数据存储路径 (支持 .parquet)')
    '''
    # 历史特征：根据实际列名增减
    parser.add_argument('--past_features', type=str, nargs='*',
                    default=['wd_deg', 'SWDR_W', 'max_wv', 'wv_m', 'rho_g', 'max_PAR', 'VPdef_mbar', 'PAR_ol', 'VPmax_mbar', 'rh', 'Tpot_K'],
                    help='历史特征列表')
    '''
    parser.add_argument('--past_features', type=str, nargs='*',
                    default=['p_mbar', 'Tdew_degC', 'VPact_mbar', 'H2OC_m', 'sh_g', 'SWDR_W', 'PAR_ol', 'max_PAR', 'rho_g', 'Tlog_degC', 'VPmax_mbar', 'T_degC', 'Tpot_K', 'wd_deg', 'max_wv', 'wv_m', 'VPdef_mbar', 'rh', 'rain_mm', 'raining_s'],
                    help='历史特征列表')
    
    # 未来特征：日期衍生特征
    parser.add_argument('--forward_features', type=str, nargs='*',
                        # default=['month', 'year'],
                        default=[],
                        help='未来特征列表')
    parser.add_argument('--target', type=str, default='OT',
                    help='预测目标字段')

    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=4,
                        help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM堆叠层数')
    parser.add_argument('--embedding_size', type=int, default=12,
                        help='嵌入层维度')
    parser.add_argument('--attn_head', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--hidden_size_target', type=int, default=4,
                        help='因果模型目标侧LSTM隐藏层维度')
    parser.add_argument('--attn_head_target', type=int, default=8,
                        help='因果模型目标侧注意力头数')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.015892,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=6.25e-05,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=36,
                        help='训练轮数')
    parser.add_argument('--batch_num', type=int, default=12,
                        help='批量大小')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='批量大小')
    parser.add_argument('--sequence_length', type=int, default=96,
                        help='输入序列长度')
    parser.add_argument('--step_forward', type=int, default=96,
                        help='预测步长')
    parser.add_argument('--train_rate', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--valid_rate', type=float, default=0.1,
                        help='训练集比例')
    parser.add_argument('--normal_epochs', type=int, default=5,
                        help='因果模型正常训练轮数')
    parser.add_argument('--adv_epochs', type=int, default=5,
                        help='因果模型对抗训练轮数')
    parser.add_argument('--adv_weight', type=float, default=0.001,
                        help='因果模型对抗损失权重')
    parser.add_argument('--nor_weight', type=float, default=1,
                        help='主任务损失权重（目前没有启用）')
    parser.add_argument('--lr_adv', type=float, default=0.005,
                        help='因果模型对抗优化器学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout_rate')

    # ID Embedding 维度 (公司数量可能较多，适当调大)
    parser.add_argument('--id_embedding_size', type=int, default=10, help='ID嵌入层维度')
    # 布尔参数
    parser.add_argument('--model', type=str, default='LSTMCausalAd',
                        help='模型选择')
    parser.add_argument('--fix_seed', type=bool, default=True,
                        help='是否固定随机种子')
    parser.add_argument('--share_outNet', type=bool, default=True,
                        help='因果模型的encoder和decoder是否共享输出层')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='是否保存模型参数')

    # 实验参数
    parser.add_argument('--num_runs', type=int, default=10,
                        help='实验轮数')
    parser.add_argument('--search_count', type=int, default=64,
                        help='超参数搜索轮数')
    parser.add_argument('--config_path', type=str, default='./best_params.yaml',
                        help='最佳参数存储路径')
    parser.add_argument('--save_path', type=str, default='./save_model/',
                        help='最佳参数存储路径')

    # 早停
    parser.add_argument('--patience', type=int, default=20,
                        help='早停的忍耐次数')

    # === TimeMixer 专用参数 ===
    parser.add_argument('--d_ff', type=int, default=32, help='d_ff')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='down_sampling_layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down_sampling_window')
    parser.add_argument('--channel_independence', type=int, default=0, help='channel_independence')
    
    return parser


def load_model_config(args):
    with open(args.config_path) as f:
        configs = yaml.safe_load(f)

    common = configs.get('common', {})

    model_name = args.model
    if args.model in ['LSTMPostMarkAttCausalAd', 'LSTMCausalAd']:
        model_name = f'{args.model}_{args.share_outNet}'

    model = configs.get(model_name, {})

    config = {**common, **model}

    # 用配置覆盖args对象
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args
