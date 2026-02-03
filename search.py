import gc
import os
import ast
import argparse
import json

import optuna
import pandas as pd
import torch
import polars as pl

from config import create_parser, load_model_config
from evaluate import evaluate_model
from train import train_model
from util import *


# 定义目标函数
def objective(trial):
    set_seed()

    try:
        # 公共参数
        args.embedding_size = trial.suggest_int('embedding_size', 4, 12, step=4)  # 嵌入层维度
        args.hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32])  # LSTM隐藏层维度
        args.lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)  # 学习率
        args.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)  # 权重衰减
        args.id_embedding_size = trial.suggest_int('id_embedding_size', 4, 16, step=2)  # ID嵌入层维度

        if args.model == 'LSTM':
            pass  # 只有公共参数

        elif args.model == 'LSTM_Attention':
            args.attn_head = trial.suggest_categorical('attn_head', [1, 2, 4])  # 注意力头数

        elif args.model == 'LSTMPostMarkAttCausalAd':
            args.attn_head = trial.suggest_categorical('attn_head', [1, 2, 4])
            args.attn_head_target = trial.suggest_categorical('attn_head_target', [1, 2, 4])
            args.hidden_size_target = trial.suggest_categorical('hidden_size_target', [4, 8, 16, 32])
            args.normal_epochs = trial.suggest_int('normal_epochs', 3, 9, step=1)
            args.adv_weight = trial.suggest_float('adv_weight', 1e-5, 1e-1, log=True)
            args.lr_adv = trial.suggest_float('lr_adv', 1e-3, 1e-1, log=True)

        elif args.model == 'LSTMCausalAd':
            args.hidden_size_target = trial.suggest_categorical('hidden_size_target', [4, 8, 16, 32])
            args.normal_epochs = trial.suggest_int('normal_epochs', 4, 10, step=1)
            args.adv_weight = trial.suggest_float('adv_weight', 1e-5, 1e-1, log=True)
            args.lr_adv = trial.suggest_float('lr_adv', 1e-3, 1e-1, log=True)

        else:
            raise ValueError('model name error')

        model, use_adversarial = create_model(args)
        model = model.to(device)

        print("初始模型指纹:", model_fingerprint(model))

        model, history = train_model(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            scaler=scaler,
            args=args,
            use_adversarial=use_adversarial
        )

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
    
        # print(f"Run {num_run + 1}/{args.num_runs}")
        # print(f"Train RMSE: {train_rmse_original:.4f} | Test RMSE: {test_rmse_original:.4f}")
    
        print(f"\n最终测试结果（标准化数据）:")
        print(f"训练集 RMSE: {train_rmse_original:.4f}")
        print(f"训练集 MSE: {train_metrics_original['mse']:.4f}")
        print(f"训练集 MAE: {train_metrics_original['mae']:.4f}")
        
        print(f"测试集 RMSE: {test_rmse_original:.4f}")
        print(f"测试集 MSE: {test_metrics_original['mse']:.4f}")
        print(f"测试集 MAE: {test_metrics_original['mae']:.4f}")
        
        # print(f"Train R²: {train_metrics_original['mae']:.4f} | Test R²: {test_metrics_original['r2']:.4f}")

        return test_rmse_original
        
    finally:
        del model
        torch.cuda.empty_cache()  # 清除PyTorch CUDA缓存
        gc.collect()  # 调用Python垃圾回收


if __name__ == '__main__':
    # 创建研究对象
    parser = create_parser()
    args = parser.parse_args()
    # args = load_model_config(args)
    
    # 调试：先打印原始值
    print(f"接收到的原始参数: {args.past_features}")
    print(f"参数类型: {type(args.past_features)}")
    
    '''
    # 尝试多种解析方式
    features = None
    
    # 方法1: 尝试直接解析JSON
    try:
        features = json.loads(args.past_features)
        print("使用JSON解析成功")
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        # 方法2: 尝试用ast.literal_eval
        try:
            features = ast.literal_eval(args.past_features)
            print("使用ast.literal_eval解析成功")
        except Exception as e2:
            print(f"ast.literal_eval解析失败: {e2}")
            # 方法3: 清理字符串
            try:
                # 去除可能的额外引号和方括号
                cleaned = args.past_features.strip()
                if cleaned.startswith('"[') and cleaned.endswith(']"'):
                    cleaned = cleaned[2:-2]  # 去掉外层的双引号和方括号
                elif cleaned.startswith("'[") and cleaned.endswith("]'"):
                    cleaned = cleaned[2:-2]
                
                # 尝试解析清理后的字符串
                features = ast.literal_eval(f'[{cleaned}]')
                print("通过清理字符串解析成功")
            except Exception as e3:
                print(f"所有解析方法都失败: {e3}")
                exit(1)
    
    if features is None or len(features) == 0:
        print("错误: 无法解析特征列表或特征列表为空")
        exit(1)
    
    print(f"解析后的特征数量: {len(features)}")
    print(f"特征列表: {features}")
    '''

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
    # 根据目标列名自动判断使用哪个处理函数
    if args.target == 'ret':
        train_data, valid_data, test_data, scaler, agent_encoder, num_ids = prepare_financial_data(
            data=data,
            features=features,
            target=target,
            sequence_length=args.sequence_length,
            step_forward=args.step_forward,
            train_rate=args.train_rate,
            valid_rate=args.valid_rate,
            is_scaler = False,
            use_cache = True,
            is_save_cache = True,
            use_rank_norm=args.use_rank_norm
        )
    if args.target == 'OT':
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
        
    else:
        # 原有的逻辑
        train_data, valid_data, test_data, scaler, agent_encoder, num_ids = prepare_team_data(
            data=data,
            features=features,
            target=target,
            sequence_length=args.sequence_length,
            step_forward=args.step_forward,
            train_rate=args.train_rate,
            valid_rate=args.valid_rate
        )
    
    args.num_ids = num_ids
    
    print(f'train_data_len:{len(train_data)}')
    print(f'valid_data_len:{len(valid_data)}')
    print(f'test_data_len:{len(test_data)}')

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

    study = optuna.create_study(
        study_name=f"{args.model}_{args.input_size}",  # 唯一标识
        storage=f"sqlite:///optuna/optuna_{args.model}_{args.input_size}_{args.share_outNet}.db",  # 持久化存储
        direction="minimize",  # 目标是最小化误差 maximize(R2) minimize(MSE)
        load_if_exists=False,  # 不加载已有Study
        sampler=optuna.samplers.TPESampler(seed = 2026)
    )

    # 计算剩余需要运行的试验次数
    completed_trials = len(study.trials)
    remaining_trials = max(0, args.search_count - completed_trials)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
        print(f'Best trial: {study.best_trial.value}')
        print(f'Best parameters: {study.best_trial.params}')

        study.enqueue_trial(study.best_params)  # 重新注入参数
        study.optimize(objective, n_trials=1)  # 验证结果是否一致
    else:
        print(f"已经完成{args.search_count}次试验，无需继续优化")
