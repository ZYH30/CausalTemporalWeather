import gc
import os

import joblib
import polars as pl

from config import create_parser, load_model_config
from evaluate import evaluate_model
from train import train_model
from util import *


def main():
    if args.fix_seed:
        set_seed()

    model, use_adversarial = create_model(args)

    model = model.to(device)
    model, history = train_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        scaler=scaler,
        args=args,
        use_adversarial=use_adversarial
    )

    if args.save_model:
        os.makedirs(args.save_path, exist_ok=True)

        model_name = f"{args.model}_{args.fix_seed}_{num_run}.pth"

        # 保存模型权重
        model_path = os.path.join(args.save_path, model_name)
        torch.save(model.state_dict(), model_path)

        print(f"模型已保存到: {model_path}")

        scalar_path = os.path.join(args.save_path, 'agent_scaler.pkl')
        # 保存标准化器
        joblib.dump(scaler, scalar_path)

    _, _, train_rmse_original, train_metrics_original, _ = evaluate_model(
        model=model,
        test_data=train_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scaler=scaler,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Train',
        plot=False,
        is_original=True,
        is_scaler = True
    )

    _, _, test_rmse_original, test_metrics_original, df_result = evaluate_model(
        model=model,
        test_data=test_data,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        scaler=scaler,
        batch_size=args.batch_size,
        past_input_size=args.past_input_size,
        forward_input_size=args.forward_input_size,
        mode='Test',
        plot=False,
        is_original=True,
        is_scaler = False
    )

    print(f"Run {num_run + 1}/{args.num_runs}")
    print(f"Train RMSE: {train_rmse_original:.4f} | Test RMSE: {test_rmse_original:.4f}")

    print(f"\n最终测试结果（原始数据尺度）:")
    print(f"测试集 RMSE: {test_rmse_original:.4f}")
    print(f"测试集 MSE: {test_metrics_original['mse']:.4f}")
    print(f"测试集 MAE: {test_metrics_original['mae']:.4f}")
    print(f"测试集 R²: {test_metrics_original['r2']:.4f}")
    print(f"测试集 R²: {test_metrics_original['r2_standard']:.4f}")
    
    return train_metrics_original, test_rmse_original, df_result


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args = load_model_config(args)

    # --- 修改数据读取部分 ---
    print(f"Loading data from: {args.data_path}")
    if args.data_path.endswith('.parquet'):
        # 使用 Polars 读取 Parquet 并转为 Pandas
        data = pl.read_parquet(args.data_path).to_pandas()
    else:
        # 兼容旧 CSV 逻辑
        data = load_data(args.data_path)
    
    # 重构特征字典
    features = {
        'forward_feature': args.forward_features,
        'past_feature': args.past_features
    }
    target = args.target
    
    # --- 修改数据准备调用部分 ---
    train_data, valid_data, test_data, scaler, agent_encoder, num_ids = prepare_weather_data(
        data=data,
        features=features,
        target=target,
        sequence_length=args.sequence_length,
        step_forward=args.step_forward,
        train_rate=args.train_rate,
        valid_rate=args.valid_rate,
        is_scaler = True,
        use_cache = False,
        is_save_cache = True,
        modelName = args.model
    )
    
    args.num_ids = num_ids
    
    print(f'train_data_len:{len(train_data)}')
    print(f'valid_data_len:{len(valid_data)}')
    print(f'test_data_len:{len(test_data)}')
    '''
    # 初始化模型参数
    args.forward_input_size = len(features['forward_feature'])
    args.past_input_size = len(features['past_feature'])
    args.input_size = args.forward_input_size + args.past_input_size
    '''
    # ==========================================
    # --- 修复：从数据中自动推断真实输入维度 ---
    # ==========================================
    if len(train_data) > 0:
        # 获取第一个样本: (past, forward, target, ...)
        sample_past, sample_forward = train_data[0][0], train_data[0][1]
        
        # 动态获取维度
        real_past_dim = sample_past.shape[1] if sample_past is not None else 0
        real_forward_dim = sample_forward.shape[1] if sample_forward is not None else 0
        
        # 更新参数
        args.past_input_size = real_past_dim
        args.forward_input_size = real_forward_dim
        args.input_size = real_past_dim + real_forward_dim
        
        print(f"✅ 维度自动校正: Past={real_past_dim}, Forward={real_forward_dim}, Total={args.input_size}")
        
        # 双重检查
        config_dim = len(features['past_feature']) + len(features['forward_feature'])
        if args.input_size != config_dim:
            print(f"⚠️ 警告: 实际数据维度 ({args.input_size}) 与配置维度 ({config_dim}) 不一致。")
            print(f"   原因: 预处理阶段可能删除了 {config_dim - args.input_size} 个高缺失率特征 (这是正常的)。")
    else:
        raise ValueError("训练集为空，无法推断模型维度！")
    # ==========================================
    
    rmses = []
    if args.fix_seed:
        args.num_runs = 1
    for num_run in range(args.num_runs):
        _, rmse, df_result = main()
        rmses.append(rmse)
    
        # 释放显存
        torch.cuda.empty_cache()  # 清除PyTorch CUDA缓存
        gc.collect()  # 调用Python垃圾回收
    
    if args.num_runs > 1:
        compute_and_log_statistics(rmses)
