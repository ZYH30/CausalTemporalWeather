import json
import os

import joblib
from sklearn.metrics import mean_squared_error

from config import *
from util import *


def prepare_data(data, features, target, scalar_y, sequence_length=30, step_forward=7):
    scaled_target = scalar_y.fit_transform(data[[target]]).astype('float32')
    scalar_x = StandardScaler()
    if len(features['forward_feature']) > 0 and len(features['past_feature']) > 0:
        scalar_x.fit(data[features['past_feature'] + features['forward_feature']])
    elif len(features['forward_feature']) > 0:
        scalar_x.fit(data[features['forward_feature']])
    elif len(features['past_feature']) > 0:
        scalar_x.fit(data[features['past_feature']])

    sequences = []
    for agent, group in data.groupby('agent_name'):
        group = group.sort_values('date_id').reset_index(drop=True)
        scaled_features_past = scalar_x.transform(group[features['past_feature']]).astype('float32')
        scaled_features_forward = scalar_x.transform(group[features['forward_feature']]).astype('float32')
        for i in range(len(group) - sequence_length - step_forward + 1):
            seq_target = scaled_target[i: i + sequence_length + step_forward]
            # 参考时间 = history 最后一条数据日期
            ref_date = group.loc[i + sequence_length - 1, 'date_id']

            if len(features['forward_feature']) > 0 and len(features['past_feature']) > 0:
                seq_features_past = scaled_features_past[i: i + sequence_length]
                seq_features_forward = scaled_features_forward[i: i + sequence_length + step_forward]
                sequences.append((seq_features_past, seq_features_forward, seq_target, ref_date, agent))

            elif len(features['forward_feature']) > 0:
                seq_features_forward = scaled_features_forward[i: i + sequence_length + step_forward]
                sequences.append((None, seq_features_forward, seq_target, ref_date, agent))

            elif len(features['past_feature']) > 0:
                seq_features_past = scaled_features_past[i: i + sequence_length]
                sequences.append((seq_features_past, None, seq_target, ref_date, agent))

            else:
                sequences.append((None, None, seq_target, ref_date, agent))

    return sequences


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
        batch_ref_dates = []
        batch_agents = []

        # 按索引顺序处理当前批次的样本
        for i in batch_indices:
            if forward_input_size > 0:
                if past_input_size > 0:
                    past, forward, target, ref_date, agent = data[i]
                    batch_past.append(past)
                    batch_forward.append(forward)
                else:
                    forward, target, ref_date, agent = data[i]
                    batch_forward.append(forward)
            else:
                past, target, ref_date, agent = data[i]
                batch_past.append(past)

            batch_target.append(target)
            batch_ref_dates.append(ref_date)
            batch_agents.append(agent)

        # 转换为numpy数组
        batch_past = np.array(batch_past) if batch_past else None
        batch_forward = np.array(batch_forward) if batch_forward else None
        batch_target = np.array(batch_target)
        batch_ref_dates = np.array(batch_ref_dates)
        batch_agents = np.array(batch_agents)

        # 更新索引
        start_idx = end_idx

        yield batch_past, batch_forward, batch_target, batch_ref_dates, batch_agents


def evaluate_period(
        model,
        data,
        batch_size,
        past_input_size,
        forward_input_size,
        sequence_length,
        step_forward,
        scaler_y
):
    """
    单样本预测，每条样本独立标准化历史输入特征。
    预测部分保持原始尺度，计算每条样本RMSE。

    data: iterable of (past, forward, label, ref_date, agent)
    """
    model.eval()
    results = []

    for batch_idx, (past_feats, forward_feats, targets, ref_dates, agents) in enumerate(
            batch_generator(
                data,
                batch_size=batch_size,
                past_input_size=past_input_size,
                forward_input_size=forward_input_size,
                shuffle=False
            ),
            start=1
    ):
        x = numpy_to_tensor(past_feats)
        x_forward = numpy_to_tensor(forward_feats)
        y = numpy_to_tensor(targets)
        # 模型预测
        with torch.no_grad():
            _, outputs = model(X=x, y=y, X_forward=x_forward, is_training=False)

        # 处理输出
        predictions, actuals = process_outputs(
            outputs,
            y,
            sequence_length,
            step_forward
        )

        # 逆标准化
        predictions_r = scaler_y.inverse_transform(predictions)
        actuals_r = scaler_y.inverse_transform(actuals)

        predictions_r = predictions_r.reshape(outputs.shape[0], -1, 1)
        actuals_r = actuals_r.reshape(outputs.shape[0], -1, 1)

        for pred, label, ref_date, agent in zip(predictions_r, actuals_r, ref_dates, agents):
            sample_rmse = np.sqrt(mean_squared_error(pred, label))
            results.append({
                "agent": agent,
                "ref_date": ref_date,
                "rmse": sample_rmse
            })
            print(f"agent={agent}, ref_date={ref_date}, rmse={sample_rmse}")

    return results


def numpy_to_tensor(data: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    """将numpy数组转换为tensor，处理None情况"""
    if data is None:
        return None
    return torch.from_numpy(data).float().to(device)


def process_outputs(
        outputs: torch.Tensor,
        target_tensor: torch.Tensor,
        sequence_length: int,
        step_forward: int
):
    """处理模型输出和目标值"""
    current_batch_size = target_tensor.shape[0]
    required_elements = current_batch_size * step_forward

    # 处理预测值
    predictions = outputs.cpu().detach().view(required_elements, -1).numpy()

    # 处理真实值
    if step_forward > 1:
        actuals = target_tensor[:, sequence_length:, :].reshape(required_elements, -1).cpu().numpy()
    else:
        actuals = target_tensor[:, sequence_length: sequence_length + 1, :].reshape(required_elements, -1).cpu().numpy()

    return predictions, actuals


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = load_model_config(args)
    print_args(args)

    data = load_data(args.data_path)
    data = data[:1000]

    # 重构特征字典
    features = {
        'forward_feature': args.forward_features,
        'past_feature': args.past_features
    }
    target = args.target

    # 初始化模型参数
    args.forward_input_size = len(features['forward_feature'])
    args.past_input_size = len(features['past_feature'])
    args.input_size = args.forward_input_size + args.past_input_size

    scalar_path = os.path.join(args.save_path, 'scalar_y.pkl')
    scalar_y = joblib.load(scalar_path)

    data = prepare_data(
        data=data,
        features=features,
        target=target,
        scalar_y=scalar_y,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward
    )
    print(f'num of samples:{len(data)}')

    if args.fix_seed:
        set_seed()

    model, use_adversarial = create_model(args)
    model = model.to(device)

    model_name = f"{args.model}_{args.fix_seed}_{0}.pth"

    # 加载模型权重
    model_path = os.path.join(args.save_path, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = evaluate_period(
        model=model,
        data=data,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scaler_y=scalar_y
    )

    # 保存为JSON格式
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已保存为 evaluation_results.json")


if __name__ == '__main__':
    main()