from typing import Optional, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error,mean_squared_error
import pandas as pd

from util import device, batch_generator, AgentStandardScaler, _to_tensor


# 【新增】插入这个函数
def huber_loss_score(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss)
    
def _process_outputs(
        outputs: torch.Tensor,
        target_tensor: torch.Tensor,
        sequence_length: int,
        step_forward: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理模型输出和目标值，保留 batch/step 结构
    """
    # outputs shape: (batch, step_forward, feature_dim)
    predictions = outputs.cpu().detach().numpy()

    # target shape: (batch, total_seq_len, feature_dim)
    actuals = target_tensor[:, sequence_length: sequence_length + step_forward, :].cpu().numpy()

    return predictions, actuals


def plot_predictions_vs_actuals(all_predictions, all_actuals, step_forward, mode='valid', 
                                max_samples=1000, sample_interval=10, figsize=(15, 10)):
    """
    Visualize predictions vs actual values
    
    Parameters:
        all_predictions: Prediction array (n_samples, step_forward) or (n_samples, 1)
        all_actuals: Actual value array (n_samples, step_forward) or (n_samples, 1)
        step_forward: Number of prediction steps
        mode: Dataset mode ('train', 'valid', 'test')
        max_samples: Maximum number of samples to display
        sample_interval: Sampling interval
        figsize: Figure size
    """
    # Ensure data is 2D array
    all_predictions = np.atleast_2d(all_predictions)
    all_actuals = np.atleast_2d(all_actuals)
    
    n_samples, output_dim = all_predictions.shape
    
    # If output dimension is 1 but step_forward>1, adjust processing
    if output_dim == 1 and step_forward > 1:
        print(f"Note: Model output dimension is 1, but prediction steps is {step_forward}. Treating as single-step prediction.")
    
    # If too many samples, downsample
    if n_samples > max_samples:
        indices = np.arange(0, n_samples, sample_interval)
        if len(indices) > max_samples:
            indices = indices[:max_samples]
        all_predictions = all_predictions[indices]
        all_actuals = all_actuals[indices]
        n_samples = len(indices)
        print(f"Downsampling: Showing {len(indices)} out of {n_samples} samples")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Predictions vs Actuals ({mode} Set)', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Predictions vs Actuals
    ax1 = axes[0, 0]
    ax1.scatter(all_actuals.ravel(), all_predictions.ravel(), 
               alpha=0.6, s=10, c='blue', edgecolors='white', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(all_actuals.min(), all_predictions.min())
    max_val = max(all_actuals.max(), all_predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    
    rmse = root_mean_squared_error(all_actuals, all_predictions)
    ax1.set_title(f'Predictions vs Actuals (RMSE: {rmse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(all_actuals.ravel(), all_predictions.ravel())[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Error distribution by prediction step
    ax2 = axes[0, 1]
    
    if output_dim > 1:
        # Multi-step prediction: Box plot for each step
        errors_by_step = []
        for step in range(min(output_dim, step_forward)):
            step_errors = all_predictions[:, step] - all_actuals[:, step]
            errors_by_step.append(step_errors)
        
        box = ax2.boxplot(errors_by_step, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', color='blue'),
                         medianprops=dict(color='red', linewidth=2))
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Prediction Step')
        ax2.set_ylabel('Prediction Error (Predicted - Actual)')
        ax2.set_title(f'Error Distribution by Prediction Step ({len(errors_by_step)} steps)')
        ax2.set_xticklabels([f'Step {i+1}' for i in range(len(errors_by_step))])
    else:
        # Single-step prediction: Single error distribution
        errors = all_predictions.ravel() - all_actuals.ravel()
        box = ax2.boxplot([errors], patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', color='blue'),
                         medianprops=dict(color='red', linewidth=2),
                         widths=0.6)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Prediction Error (Predicted - Actual)')
        ax2.set_title('Prediction Error Distribution')
        ax2.set_xticklabels(['Single-step'])
    
    ax2.grid(True, alpha=0.3)
    
    # 3. Time series comparison for random samples
    ax3 = axes[1, 0]
    n_series_to_plot = min(5, n_samples)
    
    if n_series_to_plot > 0:
        random_indices = np.random.choice(n_samples, n_series_to_plot, replace=False)
        
        if output_dim > 1:
            # Multi-step prediction: Plot each prediction step
            for idx, sample_idx in enumerate(random_indices):
                steps = np.arange(1, output_dim + 1)
                ax3.plot(steps, all_actuals[sample_idx], 'o-', linewidth=2, 
                        alpha=0.8, markersize=6)
                ax3.plot(steps, all_predictions[sample_idx], 's--', linewidth=2, 
                        alpha=0.8, markersize=6)
            
            # Add mean lines
            if len(random_indices) > 1:
                mean_actual = np.mean(all_actuals[random_indices], axis=0)
                mean_pred = np.mean(all_predictions[random_indices], axis=0)
                ax3.plot(steps, mean_actual, 'b-', linewidth=3, label='Actual Mean', alpha=0.8)
                ax3.plot(steps, mean_pred, 'r--', linewidth=3, label='Predicted Mean', alpha=0.8)
            
            ax3.set_xlabel('Prediction Step')
        else:
            # Single-step prediction: Bar chart comparison
            for idx, sample_idx in enumerate(random_indices):
                ax3.bar(idx - 0.2, all_actuals[sample_idx, 0], width=0.4, 
                       alpha=0.7, label='Actual' if idx == 0 else "")
                ax3.bar(idx + 0.2, all_predictions[sample_idx, 0], width=0.4, 
                       alpha=0.7, label='Predicted' if idx == 0 else "")
            
            ax3.set_xlabel('Sample Index')
            ax3.set_xticks(range(n_series_to_plot))
            ax3.set_xticklabels([f'Sample {i+1}' for i in range(n_series_to_plot)])
        
        ax3.set_ylabel('Value')
        ax3.set_title(f'Prediction Comparison for {n_series_to_plot} Random Samples')
        ax3.legend(loc='best', fontsize='small')
    else:
        ax3.text(0.5, 0.5, 'No samples to display', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Prediction Comparison')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution histogram
    ax4 = axes[1, 1]
    errors = all_predictions.ravel() - all_actuals.ravel()
    ax4.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add normal distribution fit
    try:
        from scipy.stats import norm
        mu, std = norm.fit(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        p = norm.pdf(x, mu, std)
        ax4.plot(x, p, 'r-', linewidth=2, label=f'N(μ={mu:.4f}, σ={std:.4f})')
    except ImportError:
        print("Note: scipy not installed, skipping normal distribution fit")
    
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Error Distribution (Mean={errors.mean():.4f}, Std={errors.std():.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def evaluate_model(
        model: torch.nn.Module,
        test_data: List,
        sequence_length: int,
        step_forward: int,
        scaler,
        batch_size: int,
        past_input_size: int = 0,
        forward_input_size: int = 0,
        mode='valid',
        verbose=True,
        plot=False,
        plot_kwargs=None,
        is_original=False,
        is_scaler = False,
        return_loos_fn = 'MSE'
):
    """
    评估模型在测试集上的表现，先逆标准化，再展平计算全局指标。
    当 is_original=True 时，额外返回详细预测结果的 DataFrame。
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    # 用于构建 DataFrame 的列表
    all_permnos = []
    all_dates = []
    all_original_targets = [] 

    # 循环遍历 Batch
    for batch_idx, (past, forward, target, timestamps, agent_ids, permno_vals, original_target, x_mark) in enumerate(
            batch_generator(
                test_data,
                batch_size=batch_size,
                past_input_size=past_input_size,
                forward_input_size=forward_input_size,
                shuffle=False
            ),
            start=1
    ):
        past_tensor = _to_tensor(past)
        forward_tensor = _to_tensor(forward)
        target_tensor = _to_tensor(target)
        
        if agent_ids is not None:
            agent_ids = torch.LongTensor(agent_ids).to(device)

        # 2. === [Bug 修复核心] 强制转换 x_mark 为 Tensor ===
        if x_mark is not None:
            # 必须转换为 FloatTensor 并移至 device
            x_mark = torch.FloatTensor(x_mark).to(device)
                
        # 在 evaluate.py 中找到这一行
        # _, outputs = model(X=past_tensor, y=target_tensor, ...)
        
        '''
        # === 修改开始 ===
        # 1. 强制深度拷贝 Target
        blind_target = target_tensor.clone()
        
        # 2. "致盲"：将未来部分的数据全部强制置为 0 (或者随机噪声)
        # 注意：TargetTower 只需要前 sequence_length 个点作为历史
        blind_target[:, sequence_length:, :] = 0.0 
        
        # 3. 将致盲后的数据传给模型
        # 务必保持 is_training=False，否则模型会把 0 当作真实标签去学习
        with torch.no_grad():
            _, outputs = model(
                X=past_tensor, 
                y=blind_target,  # <--- 传入致盲数据
                X_forward=forward_tensor,
                ids=agent_ids,
                is_training=False 
            )
        # === 修改结束 ===
        '''

        # === [修改点 C] 构建动态参数字典 ===
        # 基础参数（所有模型通用）
        forward_kwargs = {
            'X': past_tensor,
            'y': target_tensor, # 评估模式下有些模型可能需要 y 做 Teacher Forcing 的占位，或者计算 Loss
            'X_forward': forward_tensor,
            'ids': agent_ids,
            'is_training': False # 评估模式
        }
        
        # 仅对 TimeMixer 注入 x_mark
        # 注意：这里需要一种方式判断模型类型。
        # 通常 model 对象没有 .model_name 属性，除非我们在 Adapter 里加了，或者通过类名判断
        # 比较稳妥的方式是：尝试判断 model 是否接受 x_mark，或者直接检查类名
        # 由于 TimeMixerAdapter 是我们在 models/time_mixer_adapter.py 定义的
        if 'TimeMixerAdapter' in model.__class__.__name__ and x_mark is not None:
             forward_kwargs['x_mark'] = x_mark

        # 模型推理
        # 注意：evaluate 时通常只需要 outputs
        with torch.no_grad():
            _, outputs = model(**forward_kwargs)

        '''
        with torch.no_grad():
            _, outputs = model(
                X=past_tensor,
                y=target_tensor,
                X_forward=forward_tensor,
                ids=agent_ids,
                is_training=False
            )
        '''
        # predictions shape: (batch_size, step_forward, feature_dim)
        # actuals 已经被切片，只包含未来部分
        predictions, actuals = _process_outputs(outputs, target_tensor, sequence_length, step_forward)

        if is_scaler:
            predictions_r = scaler.inverse_transform_target(predictions)
            actuals_r = scaler.inverse_transform_target(actuals)
        else:
            predictions_r = predictions
            actuals_r = actuals

        # 调试：打印形状
        if verbose and batch_idx == 1:
            print(f"Batch 1 形状检查:")
            print(f"  predictions_r shape: {predictions_r.shape}")  # (batch_size, step_forward, 1)
            print(f"  actuals_r shape: {actuals_r.shape}")  # (batch_size, step_forward, 1)
            print(f"  original_target shape: {original_target.shape}")  # (batch_size, seq_len+step_forward, 1)
            
        # 展平成 (batch * step_forward, feature_dim)
        predictions_r_flat = predictions_r.reshape(-1, predictions_r.shape[-1])
        actuals_r_flat = actuals_r.reshape(-1, actuals_r.shape[-1])
        
        if verbose and batch_idx == 1:
            print(f"Batch 1 形状检查:")
            print(f"  predictions_r_flat shape: {predictions_r_flat.shape}")  # (batch_size, step_forward, 1)
            print(f"  actuals_r_flat shape: {actuals_r_flat.shape}")  # (batch_size, step_forward, 1)
            
        all_predictions.append(predictions_r_flat)
        all_actuals.append(actuals_r_flat)

        # =========================================================
        # [关键修复] 对 original_target 进行切片，只保留未来预测部分
        # =========================================================
        if original_target is not None:
            # original_target shape: (batch, seq_len + step_forward, 1)
            # 我们需要切出与 predictions 对应的部分：[sequence_length : sequence_length + step_forward]
            original_target_future = original_target[:, sequence_length : sequence_length + step_forward, :]
            
            # 展平
            original_target_flat = original_target_future.reshape(-1, original_target_future.shape[-1])
            all_original_targets.append(original_target_flat)
            
            if verbose and batch_idx == 1:
                print(f"  original_target_future shape: {original_target_future.shape}")
                print(f"  original_target_flat shape: {original_target_flat.shape}")

        # --- 收集元数据用于 DataFrame ---
        if is_original:
            # 计算每个样本重复的次数
            # 对于多步预测，每个样本需要重复 step_forward 次
            batch_size_actual = predictions_r.shape[0]
            # steps = predictions_r.shape[1]  # 应该是 step_forward
            
            # 确保 permno_vals 和 timestamps 是 numpy 数组
            if isinstance(permno_vals, str):
                # 如果是单个字符串（天气数据），创建数组
                permno_vals_array = np.array([permno_vals] * batch_size_actual)
            else:
                permno_vals_array = np.array(permno_vals)
            
            if isinstance(timestamps, (pd.Timestamp, np.datetime64)):
                # 如果是单个时间戳，创建数组
                timestamps_array = np.array([timestamps] * batch_size_actual)
            else:
                timestamps_array = np.array(timestamps)
            
            # 重复 permno 和 date 以匹配 flatten 后的长度
            batch_permnos_repeated = np.repeat(permno_vals_array, step_forward)
            batch_dates_repeated = np.repeat(timestamps_array, step_forward)
            
            # 检查长度是否一致
            if len(batch_permnos_repeated) != len(predictions_r_flat):
                print(f"警告: Batch {batch_idx} 长度不匹配")
                print(f"  batch_permnos_repeated: {len(batch_permnos_repeated)}")
                print(f"  predictions_r_flat: {len(predictions_r_flat)}")
                print(f"  batch_size: {batch_size_actual}, steps: {steps}")
            
            all_permnos.append(batch_permnos_repeated)
            all_dates.append(batch_dates_repeated)

        if verbose:
            batch_Huber = huber_loss_score(actuals_r_flat, predictions_r_flat, delta=1.0)
            print(f"{mode} Batch {batch_idx} Huber: {batch_Huber:.4f}")

    # 合并所有批次
    all_predictions = np.concatenate(all_predictions)
    all_actuals = np.concatenate(all_actuals)
    
    # 计算指标
    metrics = calculate_metrics(all_actuals, all_predictions)

    if verbose:
        print(f"{mode} RMSE: {metrics['rmse']:.4f}")
        print(f"{mode} MSE: {metrics['mse']:.4f}")
        print(f"{mode} MAE: {metrics['mae']:.4f}")
        print(f"{mode} R² Score: {metrics['r2']:.4f}")
        print(f"{mode} Huber: {metrics['huber_loss']:.6f}")

    if plot:
        if plot_kwargs is None:
            plot_kwargs = {}
        fig = plot_predictions_vs_actuals(
            all_predictions, all_actuals, step_forward, mode=mode, **plot_kwargs
        )
        metrics['plot_figure'] = fig

    if is_original:
        # 确保 original_target 也被合并
        all_original_targets = np.concatenate(all_original_targets)
        
        # 再次检查长度一致性 (Debug用)
        assert len(all_predictions) == len(all_original_targets), \
            f"Length mismatch: Preds {len(all_predictions)} vs Original {len(all_original_targets)}"

        # 构建 DataFrame
        all_permnos = np.concatenate(all_permnos)
        all_dates = np.concatenate(all_dates)

        # 调试：打印各个数组的长度
        print(f"长度检查:")
        print(f"  all_predictions: {len(all_predictions.ravel())}")
        print(f"  all_actuals: {len(all_actuals.ravel())}")
        print(f"  all_original_targets: {len(all_original_targets.ravel())}")
        print(f"  all_permnos: {len(all_permnos)}")
        print(f"  all_dates: {len(all_dates)}")
        
        df_result = pd.DataFrame({
            'permno_val': all_permnos,
            'dates': all_dates,
            'model_target': all_actuals.ravel(),       # 训练用的目标 (Rank Score)
            'predicted_score': all_predictions.ravel(), # 预测的 Rank Score
            'original_return': all_original_targets.ravel() # 真实的原始收益率
        })
        if return_loos_fn == 'MSE':
            return all_predictions, all_actuals, metrics['rmse'], metrics, df_result
        elif return_loos_fn == 'huber':
            return all_predictions, all_actuals, metrics['huber_loss'], metrics, df_result
        else:
            print('unknown loss_fn, set MSE')
            return all_predictions, all_actuals, metrics['rmse'], metrics, df_result
    else:
        if return_loos_fn == 'MSE':
            return all_predictions, all_actuals, metrics['rmse']
        elif return_loos_fn == 'huber':
            return all_predictions, all_actuals, metrics['huber_loss']
        else:
            print('unknown loss_fn, set MSE')
            return all_predictions, all_actuals, metrics['rmse'], metrics, df_result
            

def calculate_gkx_r2_oos(y_true, y_pred):
    """
    计算 GKX (2020) 定义的样本外 R2
    特点: 分母是 sum(y_true^2)，不减去均值。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()  # 确保是一维数组
    
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    
    if denominator == 0: 
        return np.nan
    return 1 - (numerator / denominator)


def calculate_metrics(y_true, y_pred, r2_method='gkx'):
    """
    计算各种评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        r2_method: R2计算方法，可选 'standard'(sklearn) 或 'gkx'(自定义)
    """
    from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
    
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    # 避免除以零
    non_zero_mask = y_true_flat != 0
    
    # 计算基本指标
    metrics = {
        'rmse': root_mean_squared_error(y_true_flat, y_pred_flat),
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'huber_loss': huber_loss_score(y_true_flat, y_pred_flat, delta=1.0)
    }
    
    # 计算 R2，根据参数选择计算方法
    if r2_method == 'gkx':
        metrics['r2'] = calculate_gkx_r2_oos(y_true_flat, y_pred_flat)
        metrics['r2_method'] = 'gkx_oos'  # 记录使用的方法
        metrics['r2_standard'] = r2_score(y_true_flat, y_pred_flat)
    elif r2_method == 'standard':
        # 使用 sklearn 的标准 R2
        try:
            metrics['r2'] = r2_score(y_true_flat, y_pred_flat)
        except Exception as e:
            print(f"计算标准R2时出错: {e}")
            metrics['r2'] = np.nan
        metrics['r2_method'] = 'standard'
    else:
        raise ValueError(f"未知的R2计算方法: {r2_method}")
    
    # 只对非零值计算MAPE
    if non_zero_mask.any():
        metrics['mape'] = mean_absolute_percentage_error(
            y_true_flat[non_zero_mask], y_pred_flat[non_zero_mask]
        ) * 100
    else:
        metrics['mape'] = np.nan
    
    # 计算其他有用的指标
    metrics['mse'] = np.mean((y_true_flat - y_pred_flat) ** 2)  # 均方误差
    metrics['correlation'] = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else np.nan
    
    # 计算 GKX R2 的各个组成部分，用于调试
    if r2_method == 'gkx':
        y_true_np = np.array(y_true_flat)
        y_pred_np = np.array(y_pred_flat)
        metrics['gkx_numerator'] = np.sum((y_true_np - y_pred_np) ** 2)
        metrics['gkx_denominator'] = np.sum(y_true_np ** 2)
    
    return metrics

# 可选：添加更详细的时间序列可视化函数
def plot_detailed_time_series(predictions, actuals, timestamps=None, sample_indices=None, 
                              max_series=20, figsize=(15, 8)):
    """
    绘制更详细的时间序列对比图
    
    参数:
        predictions: 预测值数组
        actuals: 真实值数组
        timestamps: 时间戳数组 (可选)
        sample_indices: 要绘制的特定样本索引
        max_series: 最多绘制的序列数
        figsize: 图形大小
    """
    n_samples = predictions.shape[0]
    
    if sample_indices is None:
        # 随机选择样本
        sample_indices = np.random.choice(n_samples, min(max_series, n_samples), replace=False)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 1. 所有样本的预测vs真实对比
    ax1 = axes[0]
    for idx in sample_indices:
        steps = range(predictions.shape[1])
        if timestamps is not None:
            # 如果有时序信息，使用时间戳
            time_labels = timestamps[idx]
        else:
            time_labels = [f'Step{i+1}' for i in steps]
        
        ax1.plot(time_labels, actuals[idx], 'o-', alpha=0.6, linewidth=1.5)
        ax1.plot(time_labels, predictions[idx], 's--', alpha=0.6, linewidth=1.5)
    
    # 添加均值线
    mean_actual = np.mean(actuals, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    ax1.plot(time_labels, mean_actual, 'b-', linewidth=3, label='True Mean', alpha=0.8)
    ax1.plot(time_labels, mean_pred, 'r--', linewidth=3, label='Predict Mean', alpha=0.8)
    
    ax1.set_xlabel('timeSteps')
    ax1.set_ylabel('values')
    ax1.set_title('Detailed Times Searies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差热力图
    ax2 = axes[1]
    errors = predictions - actuals
    im = ax2.imshow(errors.T, aspect='auto', cmap='RdBu_r', alpha=0.8)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('forward Steps')
    ax2.set_title('Hot Plot of Predicted Errors(red: upper, blue: lower)')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return fig