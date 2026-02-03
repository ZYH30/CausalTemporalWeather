import random
from typing import Optional, List, Dict, Tuple

import os
import joblib
import hashlib

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models import LSTMModel, LSTMPostMarkAtt, LSTMPostMarkAttCausalAd, LSTMCausalAd, TransformerModel
from models.time_mixer_adapter import TimeMixerAdapter

from collections import defaultdict
from dataSummary import analyze_data_distribution
from sklearn.preprocessing import PowerTransformer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. 在 set_seed 函数下方，添加 time_features 辅助函数:
def time_features(dates, freq='h'):
    """
    生成时间特征 [Month, Day, Weekday, Hour]，归一化到 [-0.5, 0.5]
    专供 TimeMixer 等 Transformer 类模型使用
    """
    if isinstance(dates, pd.Series):
        dates = dates.dt
    
    month = dates.month.values
    day = dates.day.values
    weekday = dates.weekday.values
    hour = dates.hour.values
    
    # 归一化策略 (TimeMixer 标准)
    f_month = (month - 1) / 11.0 - 0.5
    f_day = (day - 1) / 30.0 - 0.5
    f_weekday = weekday / 6.0 - 0.5
    f_hour = hour / 23.0 - 0.5
    
    return np.stack([f_month, f_day, f_weekday, f_hour], axis=1).astype(np.float32)
    

def batch_generator(data, batch_size, past_input_size=0, forward_input_size=0, shuffle=False):
    """
    顺序批次数据生成器（支持shuffle）
    参数：
        data: 数据集
        batch_size: 每个批次的样本数量
        past_input_size: 历史输入特征维度
        forward_input_size: 未来输入特征维度
        shuffle: 是否打乱数据顺序
    """
    num_samples = len(data)
    indices = list(range(num_samples))

    if shuffle:
        random.shuffle(indices)  # 打乱索引顺序

    start_idx = 0

    while start_idx < num_samples:
        # 计算当前批次的结束索引
        end_idx = min(start_idx + batch_size, num_samples)

        # 获取当前批次的索引（可能是打乱后的）
        batch_indices = indices[start_idx:end_idx]

        # 初始化当前批次的存储列表
        batch_past = []
        batch_forward = []
        batch_target = []
        batch_ref_date = []
        batch_ids = []
        batch_names = []
        batch_original = [] # [新增]
        batch_x_mark = [] # [新增] 存储时间特征

        # 按索引顺序处理当前批次的样本
        for i in batch_indices:
            if forward_input_size > 0:
                if past_input_size > 0:
                    past, forward, target, ref_date, uid, name, original_target, x_mark = data[i]
                    batch_past.append(past)
                    batch_forward.append(forward)
                else:
                    _, forward, target, ref_date, uid, name, original_target, x_mark = data[i]
                    batch_forward.append(forward)
            else:
                past, _, target, ref_date, uid, name, original_target, x_mark = data[i]
                batch_past.append(past)

            batch_target.append(target)
            batch_ref_date.append(ref_date)
            batch_ids.append(uid)
            batch_names.append(name)
            batch_original.append(original_target) # [新增]
            batch_x_mark.append(x_mark) # [新增]

        # 转换为numpy数组
        batch_past = np.array(batch_past) if batch_past else None
        batch_forward = np.array(batch_forward) if batch_forward else None
        batch_target = np.array(batch_target)
        batch_ref_date = np.array(batch_ref_date)
        batch_ids = np.array(batch_ids)
        batch_names = np.array(batch_names)
        batch_original = np.array(batch_original) # [新增]
        # ... (numpy 转换逻辑保持不变) ...
        # 新增 x_mark 转换
        batch_x_mark = np.array(batch_x_mark) if len(batch_x_mark) > 0 else None

        # 更新索引
        start_idx = end_idx

        # yield batch_past, batch_forward, batch_target, batch_ref_date, batch_ids, batch_names
        yield batch_past, batch_forward, batch_target, batch_ref_date, None, batch_names, batch_original, batch_x_mark # set batch_ids as none


# 检查点：验证模型初始状态
def model_fingerprint(model):
    return sum(p.sum().item() for p in model.parameters())


def create_model(args):
    # 创建模型实例
    if args.model == 'LSTM':
        model = LSTMModel(
            input_size=args.input_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            step_forward=args.step_forward,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            dropout_rate=args.dropout_rate,
            num_ids=args.num_ids,
            id_embedding_size=args.id_embedding_size
        )
        use_adversarial = False
    elif args.model == 'LSTM_Attention':
        model = LSTMPostMarkAtt(
            input_size=args.input_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            step_forward=args.step_forward,
            att_head=args.attn_head,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            dropout_rate=args.dropout_rate
        )
        use_adversarial = False
    elif args.model == 'LSTMPostMarkAttCausalAd':
        model = LSTMPostMarkAttCausalAd(
            input_size=args.input_size,
            embedding_size=args.embedding_size,
            hidden_size_feat=args.hidden_size,
            hidden_size_target=args.hidden_size_target,
            num_layers=args.num_layers,
            step_forward=args.step_forward,
            attn_head_feat=args.attn_head,
            attn_head_target=args.attn_head_target,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            share_outNet=args.share_outNet,
            dropout_rate=args.dropout_rate
        )
        use_adversarial = True
    elif args.model == 'LSTMCausalAd':
        model = LSTMCausalAd(
            input_size=args.input_size,
            embedding_size=args.embedding_size,
            hidden_size_feat=args.hidden_size,
            hidden_size_target=args.hidden_size_target,
            num_layers=args.num_layers,
            step_forward=args.step_forward,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            share_outNet=args.share_outNet,
            dropout_rate=args.dropout_rate,
            num_ids=args.num_ids,
            id_embedding_size=args.id_embedding_size
        )
        use_adversarial = True
    elif args.model == 'TransformerModel':
        model = TransformerModel(
            input_size=args.input_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            step_forward=args.step_forward,
            past_input_size=args.past_input_size,
            forward_input_size=args.forward_input_size,
            dropout_rate=args.dropout_rate,
            num_ids=args.num_ids,
            id_embedding_size=args.id_embedding_size
        )
        use_adversarial = False
    elif args.model == 'TimeMixer':
        print("Initializing TimeMixer Adapter...")
        model = TimeMixerAdapter(args)
        use_adversarial = False
    else:
        raise ValueError('model name error')
    return model, use_adversarial


def compute_and_log_statistics(rmses):
    # 计算统计量
    stats = {
        "mean": float(np.mean(rmses)),
        "std": float(np.std(rmses)),
        "min": float(np.min(rmses)),
        "max": float(np.max(rmses)),
        "cv": (np.std(rmses) / np.mean(rmses) * 100),  # 变异系数
        "runs": rmses
    }

    # 打印统计摘要
    print("\n=== Statistical Summary ===")
    print(f"Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"Coefficient of Variation: {stats['cv']:.2f}%")


def print_args(args):
    print("运行参数配置：")
    print("-" * 40)
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name:20}: {arg_value}")
    print("-" * 40)


def plow(history, use_adversarial=True):
    if use_adversarial:
        plt.figure(figsize=(15, 12))
        # 1. 总损失图
        plt.subplot(3, 1, 1)
        plt.plot(history['total'], color='blue')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        # 2. 主任务损失图
        plt.subplot(3, 1, 2)
        plt.plot(history['y'], color='green', label='y')
        plt.legend()
        plt.title('Main Task Loss')
        plt.xlabel('Epoch')
        # 3. 对抗损失图
        plt.subplot(3, 1, 3)
        plt.plot(history['x'], color='red')
        plt.title('Adversarial Loss')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(15, 12))
        # 主任务损失图
        plt.plot(history['y'], color='green', label='y')
        plt.legend()
        plt.title('Main Task Loss')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.show()


def load_data(data_path):
    return pd.read_csv(data_path, encoding='utf-8')


def _to_tensor(data: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    """将numpy数组转换为tensor，处理None情况"""
    if data is None:
        return None
    return torch.from_numpy(data).float().to(device)

class WeatherScaler:
    """
    天气数据专用标准化器，与金融数据保持完全相同的接口
    支持三类特征的标准化和反标准化
    """

    def __init__(self):
        self.past_scaler = StandardScaler()  # 过去特征标准化器
        self.forward_scaler = StandardScaler()  # 未来特征标准化器
        self.target_scaler = StandardScaler()  # 目标值标准化器
        self.fitted = False
        self.past_features = None
        self.forward_features = None
        self.target_feature = None
        self.past_mean_ = None
        self.past_std_ = None
        self.forward_mean_ = None
        self.forward_std_ = None
        self.target_mean_ = None
        self.target_std_ = None

    def fit(self, df: pd.DataFrame,
            past_features: List[str],
            forward_features: List[str],
            target_feature: str) -> None:
        """
        拟合标准化器
        参数:
            df: 包含数据的DataFrame
            past_features: 过去特征列名列表
            forward_features: 未来特征列名列表
            target_feature: 目标特征列名
        """
        # 过去特征标准化
        if past_features:
            self.past_scaler.fit(df[past_features])
            self.past_mean_ = self.past_scaler.mean_.copy()
            self.past_std_ = self.past_scaler.scale_.copy()

        # 未来特征标准化
        if forward_features:
            self.forward_scaler.fit(df[forward_features])
            self.forward_mean_ = self.forward_scaler.mean_.copy()
            self.forward_std_ = self.forward_scaler.scale_.copy()

        # 目标值标准化
        self.target_scaler.fit(df[[target_feature]])
        self.target_mean_ = self.target_scaler.mean_.copy()
        self.target_std_ = self.target_scaler.scale_.copy()

        self.past_features = past_features
        self.forward_features = forward_features
        self.target_feature = target_feature
        self.fitted = True

    def transform_past(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """标准化过去特征"""
        assert self.fitted, "请先调用 fit() 方法"
        if not self.past_features:
            return None
        return self.past_scaler.transform(df[self.past_features]).astype('float32')

    def transform_forward(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """标准化未来特征"""
        assert self.fitted, "请先调用 fit() 方法"
        if not self.forward_features:
            return None
        return self.forward_scaler.transform(df[self.forward_features]).astype('float32')

    def transform_target(self, df: pd.DataFrame) -> np.ndarray:
        """标准化目标值"""
        assert self.fitted, "请先调用 fit() 方法"
        return self.target_scaler.transform(df[[self.target_feature]]).astype('float32')

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """反标准化目标值"""
        assert self.fitted, "请先调用 fit() 方法"
        return self.target_scaler.inverse_transform(y.reshape(-1, 1))

    def inverse_transform_past(self, X_past: np.ndarray) -> Optional[np.ndarray]:
        """反标准化过去特征"""
        assert self.fitted, "请先调用 fit() 方法"
        if not self.past_features or X_past is None:
            return None
        return self.past_scaler.inverse_transform(X_past)

    def inverse_transform_forward(self, X_forward: np.ndarray) -> Optional[np.ndarray]:
        """反标准化未来特征"""
        assert self.fitted, "请先调用 fit() 方法"
        if not self.forward_features or X_forward is None:
            return None
        return self.forward_scaler.inverse_transform(X_forward)

    def get_target_stats(self) -> dict:
        """获取目标特征的统计信息"""
        assert self.fitted, "请先调用 fit() 方法"
        return {
            'mean': self.target_mean_[0],
            'std': self.target_std_[0]
        }

    def get_past_stats(self) -> dict:
        """获取过去特征的统计信息"""
        assert self.fitted, "请先调用 fit() 方法"
        if self.past_features is None:
            return {}
        return {
            'mean': dict(zip(self.past_features, self.past_mean_)),
            'std': dict(zip(self.past_features, self.past_std_))
        }

    def get_forward_stats(self) -> dict:
        """获取未来特征的统计信息"""
        assert self.fitted, "请先调用 fit() 方法"
        if self.forward_features is None:
            return {}
        return {
            'mean': dict(zip(self.forward_features, self.forward_mean_)),
            'std': dict(zip(self.forward_features, self.forward_std_))
        }
            
def prepare_weather_data(
        data: pd.DataFrame,
        features: Dict[str, List[str]],
        target: str,
        sequence_length: int = 24,
        step_forward: int = 2,
        train_rate: float = 0.7,
        valid_rate: float = 0.2,
        is_scaler: bool = True,
        use_cache: bool = True,
        is_save_cache: bool = True,
        cache_dir: str = './data_cache',
        modelName: str = 'LSTM'
) -> Tuple[List, List, List, TeamStandardScaler, None, None]:
    """
    天气数据专用预处理函数
    适配单一时间序列，每10分钟记录一次
    """
    # --- 0. 缓存检查 ---
    if use_cache:
        suffix = "_scaler" if is_scaler else ""
        cache_file = f'{cache_dir}/weather{suffix}.pkl'
        
        print(f"cache_file: {cache_file}")
        
        if os.path.exists(cache_file):
            print(f"⚡ [Cache] Loading data from: {cache_file}")
            try:
                return joblib.load(cache_file)
            except Exception:
                print("Cache load failed, reprocessing...")
                os.makedirs(cache_dir, exist_ok=True)

    print("🚀 Starting Weather Data Preprocessing...")
    
    # --- 1. 日期处理 ---
    # 删除空值
    data = data.dropna(subset=['date', target])
    
    # 转换为datetime (假设每10分钟记录，格式灵活处理)
    try:
        # 尝试多种日期格式
        try:
            data['date_obj'] = pd.to_datetime(data['date'], format='%Y%m%d%H%M')
        except:
            try:
                data['date_obj'] = pd.to_datetime(data['date'])
            except:
                # 如果都不行，尝试转为字符串再解析
                data['date_obj'] = pd.to_datetime(data['date'].astype(str))
    except Exception as e:
        print(f"日期转换失败，请检查 date 列数据格式。示例数据: {data['date'].iloc[0]}")
        raise e
    
    # 添加时间特征
    if 'year' in features['forward_feature']:
        data['year'] = data['date_obj'].dt.year
    if 'month' in features['forward_feature']:
        data['month'] = data['date_obj'].dt.month
    if 'hour' in features['forward_feature']:
        data['hour'] = data['date_obj'].dt.hour
    if 'minute' in features['forward_feature']:
        data['minute'] = data['date_obj'].dt.minute
    
    # --- 2. 特征筛选与缺失值处理 ---
    required_cols = list(set(
        ['date_obj', target] + 
        features['past_feature'] + 
        features['forward_feature']
    ))
    
    data = data[required_cols]
    target_related_cols = [target]
    
    # 使用天气专用缺失值处理
    data, updated_past, updated_forward = handle_missing_values_weather(
        data, 
        features, 
        target_cols=target_related_cols,
        threshold=0.4
    )
    
    features['past_feature'] = updated_past
    features['forward_feature'] = updated_forward

    # === [新增关键逻辑] 特征重排: 强制 Target 到最后 ===
    # 确保 Target 存在于 past_features 中 (如果是自回归预测，通常都在)
    if modelName == 'TimeMixer':
        if target in features['past_feature']:
            # 先移除，再追加到末尾
            features['past_feature'].remove(target)
            features['past_feature'].append(target)
        else:
            # 如果 Target 不在特征里 (极其罕见)，必须加进去，否则无法利用自回归趋势
            print(f"⚠️ 警告: Target '{target}' 不在历史特征中，强制添加以适配 TimeMixer")
            features['past_feature'].append(target)
            
        print(f"⚡ [Feature Alignment] 特征顺序已重排，Target '{target}' 位于最后一位。")
        print(f"⚡ Past Features: {features['past_feature']}")
    
    
    # 确保按时间排序
    data = data.sort_values('date_obj')
    
    # --- 3. 数据集切分 (按时间) ---
    unique_dates = sorted(data['date_obj'].unique())
    train_split = int(len(unique_dates) * train_rate)
    valid_split = train_split + int(len(unique_dates) * valid_rate)

    train_dates = unique_dates[:train_split]
    valid_dates = unique_dates[train_split - sequence_length : valid_split]  # 预留 overlap
    test_dates = unique_dates[valid_split - sequence_length :]

    train_data = data[data['date_obj'].isin(train_dates)].copy()
    valid_data = data[data['date_obj'].isin(valid_dates)].copy()
    test_data = data[data['date_obj'].isin(test_dates)].copy()
    
    print(f"数据集划分: 训练集 {len(train_data)} 行, 验证集 {len(valid_data)} 行, 测试集 {len(test_data)} 行")
    
    # --- 4. 统计分析 ---
    # 这里可以调用您的统计分析函数
    # stats_results = analyze_data_distribution(
    #     train_data, valid_data, test_data, 
    #     target, features
    # )
    # print(stats_results)
    
    # --- 5. 标准化 ---
    # 天气数据不需要LabelEncoder，但为了保持接口一致，创建虚拟编码器
    # from sklearn.preprocessing import StandardScaler
    # import numpy as np
    
    scaler = WeatherScaler()
    scaler.fit(
        train_data,
        past_features=features['past_feature'],
        forward_features=features['forward_feature'],
        target_feature=target
    )

    # === [修改点 A] 生成全量时间特征 ===
    print("⚡ [Adapter] Generating Time Features")
    global_time_marks = time_features(data['date_obj'])
    
    # --- 6. 高速序列生成 (天气专用) ---
    def create_weather_sequences(subset_data):
        """天气数据专用序列生成，单一时间序列"""
        sequences = []
        has_forward = len(features['forward_feature']) > 0
        
        # 确保按时间排序
        subset_data = subset_data.sort_values('date_obj')
        
        # 提取时间索引（连续的时间点）
        dates = subset_data['date_obj'].values
        time_indices = np.arange(len(subset_data))  # 使用简单整数索引

        # === [修改点 B] 获取当前子集对应的时间特征 ===
        # 简单高效的做法：直接对当前子集的时间列再算一次
        subset_time_marks = time_features(subset_data['date_obj'])
        
        # 特征与目标
        scaled_past = scaler.transform_past(subset_data)
        scaled_forward = scaler.transform_forward(subset_data) if has_forward else None
        
        if is_scaler:
            scaled_target = scaler.transform_target(subset_data)
        else:
            scaled_target = subset_data[[target]].values.astype('float32')
        
        num_samples = len(subset_data)
        max_idx = num_samples - sequence_length - step_forward + 1
        
        if max_idx <= 0:
            print(f"警告: 数据长度 {num_samples} 不足以生成序列")
            return sequences
        
        # 构建序列
        for i in range(max_idx):
            # 索引定义
            idx_input_start = i
            idx_input_end = i + sequence_length
            idx_target_start = idx_input_end
            idx_target_end = idx_input_end + step_forward
            
            # 检查连续性（天气数据每10分钟一次）
            # 使用时间索引检查，确保是连续的10分钟间隔
            time_gap_input = time_indices[idx_input_end-1] - time_indices[idx_input_start]
            time_gap_target = time_indices[idx_target_start] - time_indices[idx_input_end-1]
            
            # 输入窗口内部应该连续
            if time_gap_input != sequence_length - 1:
                continue
            # 预测目标紧接输入窗口
            if time_gap_target != 1:
                continue
            # 预测目标窗口内部连续
            if step_forward > 1:
                if time_indices[idx_target_end-1] - time_indices[idx_target_start] != step_forward - 1:
                    continue

            # === [修改点 C] 提取当前窗口的时间特征 (覆盖 Input + Pred) ===
            # TimeMixer 需要的时间特征长度 = seq_len + pred_len (用于 Encoder 和 Decoder)
            # 对应的索引范围是 [idx_input_start : idx_target_end]
            current_time_mark = subset_time_marks[idx_input_start : idx_target_end]
            
            sequences.append((
                scaled_past[idx_input_start:idx_input_end] if scaled_past is not None else None,
                scaled_forward[idx_input_start:idx_target_end] if scaled_forward is not None else None,
                scaled_target[idx_input_start:idx_target_end],  # 只取预测期的目标
                dates[idx_input_end - 1],  # 参考日期
                0,  # 虚拟实体ID
                "weather_station",  # 虚拟标识
                subset_data[[target]].values[idx_input_start:idx_target_end],  # 原始值
                current_time_mark  # <--- [新增] 第8个元素：时间特征
            ))
        
        return sequences
    
    print("⚡ 生成天气数据序列...")
    train_seq = create_weather_sequences(train_data)
    valid_seq = create_weather_sequences(valid_data)
    test_seq = create_weather_sequences(test_data)
    
    print(f"✔ 数据准备完成: 训练集={len(train_seq)} 序列, 验证集={len(valid_seq)} 序列, 测试集={len(test_seq)} 序列")
    
    result = (train_seq, valid_seq, test_seq, scaler, None, None)
    
    # --- 7. 保存缓存 ---

    if is_save_cache:
        os.makedirs(cache_dir, exist_ok=True)
        suffix = "_scaler" if is_scaler else ""
        cache_file = f'{cache_dir}/weather{suffix}.pkl'
        
        print(f"💾 Saving cache to {cache_file}")
        joblib.dump(result, cache_file)
    
    return result

def handle_missing_values_weather(data, features, target_cols, threshold=0.4):
    """
    天气数据专用缺失值处理函数
    简化版：只处理单一序列，无需截面操作
    """
    import numpy as np
    
    # 创建深拷贝
    data = data.copy()
    print(f"开始缺失值处理... 初始维度: {data.shape}")
    
    # 1. 全局 Inf 清洗
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # 2. 列筛选
    keep_always = set(['date_obj'] + target_cols)
    missing_ratio = data.isnull().mean()
    
    cols_to_drop = [
        col for col in data.columns 
        if col not in keep_always and missing_ratio[col] > threshold
    ]
    
    if cols_to_drop:
        print(f"删除高缺失率列 (> {threshold:.0%}): {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)
    
    # 更新特征列表
    new_past_features = [f for f in features['past_feature'] if f not in cols_to_drop]
    new_forward_features = [f for f in features['forward_feature'] if f not in cols_to_drop]
    
    # 3. 强制类型清洗
    data = data.sort_values('date_obj')
    processing_cols = [c for c in data.columns if c != 'date_obj']
    
    print("正在进行深度类型清洗与填充...")
    for col in processing_cols:
        # 强制转数值
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 再次检查 Inf
        mask_inf = np.isinf(data[col])
        if mask_inf.any():
            data.loc[mask_inf, col] = np.nan

        # 如果该列全是 NaN，直接填 0
        if data[col].isnull().all():
            print(f"警告: 列 {col} 全是 NaN，已填充为 0")
            data[col] = 0.0
            continue

        if data[col].isnull().sum() == 0:
            continue
            
        # 天气数据专用填充：仅使用时间序列填充
        try:
            # 纵向填充: 前3期移动平均
            data[col] = data[col].fillna(
                data[col].shift(1).rolling(window=3, min_periods=1).mean()
            )
            '''
            # 如果还有缺失，用前后均值填充
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            '''    
            # 兜底填充: 全局 0 值
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(0.0)
                
        except Exception as e:
            print(f"列 {col} 填充失败: {e}, 强制填充 0")
            data[col] = data[col].fillna(0.0)

    # 4. 最终核查
    if np.isinf(data[processing_cols]).values.any():
        print("!!! 警告: 数据中仍存在 inf，强制替换为 0 !!!")
        data = data.replace([np.inf, -np.inf], 0.0)
        
    before_len = len(data)
    data = data.dropna()
    after_len = len(data)
    
    if before_len != after_len:
        print(f"最终清洗删除了 {before_len - after_len} 行")
        
    print(f"处理完成. 维度: {data.shape}")
    
    return data, new_past_features, new_forward_features