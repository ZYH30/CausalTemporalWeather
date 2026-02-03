import gc
import os

import joblib

from config import create_parser, load_model_config
from evaluate import evaluate_model
from util import *


def main():
    if args.fix_seed:
        set_seed()

    model, use_adversarial = create_model(args)

    model = model.to(device)
    model_name = f"{args.model}_{args.fix_seed}_{0}.pth"

    # 加载模型权重
    model_path = os.path.join(args.save_path, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    scalar_path = os.path.join(args.save_path, 'scalar_y.pkl')
    scalar_y = joblib.load(scalar_path)

    _, _, train_rmse = evaluate_model(
        model=model,
        test_data=train_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scalar=scalar_y,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Train'
    )

    _, _, valid_rmse = evaluate_model(
        model=model,
        test_data=valid_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scalar=scalar_y,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Valid'
    )

    _, _, test_rmse = evaluate_model(
        model=model,
        test_data=test_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scalar=scalar_y,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Test'
    )

    print(f"Run {num_run + 1}/{args.num_runs}")
    print(f"Train RMSE: {train_rmse:.4f} | Valid RMSE: {valid_rmse:.4f} | Test RMSE: {test_rmse: .4f}")

    return train_rmse, test_rmse


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args = load_model_config(args)
    print_args(args)

    data = load_data(args.data_path)

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

    train_data, valid_data, test_data, _ = prepare_data(
        data=data,
        features=features,
        target=target,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        train_rate=args.train_rate,
        valid_rate=args.valid_rate
    )
    print(f'train_data_len:{len(train_data)}')
    print(f'valid_data_len:{len(train_data)}')
    print(f'test_data_len:{len(test_data)}')

    rmses = []
    if args.fix_seed:
        args.num_runs = 1
    for num_run in range(args.num_runs):
        _, rmse = main()
        rmses.append(rmse)

        # 释放显存
        torch.cuda.empty_cache()  # 清除PyTorch CUDA缓存
        gc.collect()  # 调用Python垃圾回收

    if args.num_runs > 1:
        compute_and_log_statistics(rmses)