import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns  # 添加缺失的导入

def analyze_data_distribution(train_data, valid_data, test_data, target_col, features):
    """
    全面分析训练集、验证集和测试集的数据分布差异

    参数:
        train_data: 训练集DataFrame
        valid_data: 验证集DataFrame
        test_data: 测试集DataFrame
        target_col: 目标列名
        features: 特征字典 {'past_feature': [], 'forward_feature': []}

    返回:
        stats_df: 包含各数据集统计指标的DataFrame
    """
    # 合并所有特征列
    all_features = features['past_feature'] + features['forward_feature']

    # 初始化结果存储
    results = []

    # 遍历所有特征和目标变量
    for col in all_features + [target_col]:
        # 计算各数据集的统计量
        train_stats = calculate_stats(train_data[col], 'train')
        valid_stats = calculate_stats(valid_data[col], 'valid')
        test_stats = calculate_stats(test_data[col], 'test')

        # 计算数据集间的差异
        diff_train_valid = calculate_difference(train_data[col], valid_data[col])
        diff_train_test = calculate_difference(train_data[col], test_data[col])

        # 合并结果
        col_results = {
            'feature': col,
            **train_stats,
            **valid_stats,
            **test_stats,
            'train_valid_ks': diff_train_valid['ks'],
            'train_valid_wasserstein': diff_train_valid['wasserstein'],
            'train_test_ks': diff_train_test['ks'],
            'train_test_wasserstein': diff_train_test['wasserstein']
        }

        results.append(col_results)

    # 转换为DataFrame
    stats_df = pd.DataFrame(results)

    # 绘制目标变量的分布对比图
    plot_distribution_comparison(train_data[target_col], 
                                valid_data[target_col], 
                                test_data[target_col], 
                                target_col)

    return stats_df

def calculate_stats(series, prefix):
    """计算单个数据列的统计量"""
    return {
        f'{prefix}_mean': series.mean(),
        f'{prefix}_std': series.std(),
        f'{prefix}_min': series.min(),
        f'{prefix}_25%': series.quantile(0.25),
        f'{prefix}_50%': series.quantile(0.50),
        f'{prefix}_75%': series.quantile(0.75),
        f'{prefix}_max': series.max(),
        f'{prefix}_skew': series.skew(),
        f'{prefix}_kurtosis': series.kurtosis(),
        f'{prefix}_count': len(series),
        f'{prefix}_nan_count': series.isna().sum()
    }

def calculate_difference(series1, series2):
    """计算两个数据列的分布差异"""
    # KS检验
    ks_stat, ks_p = stats.ks_2samp(series1.dropna(), series2.dropna())

    # Wasserstein距离
    wasserstein = stats.wasserstein_distance(
        series1.dropna().values, 
        series2.dropna().values
    )

    return {
        'ks': ks_stat,
        'wasserstein': wasserstein
    }

def plot_distribution_comparison(train_series, valid_series, test_series, col_name):
    """绘制三个数据集的分布对比图"""
    plt.figure(figsize=(15, 10))

    # 创建子图布局
    plt.subplot(2, 1, 1)
    # 核密度估计图
    sns.kdeplot(train_series, label='Train', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(valid_series, label='Validation', color='green', fill=True, alpha=0.3)
    sns.kdeplot(test_series, label='Test', color='red', fill=True, alpha=0.3)

    plt.title(f'Distribution Comparison of {col_name}', fontsize=14)
    plt.xlabel(col_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)

    # 箱线图
    plt.subplot(2, 1, 2)
    data = pd.concat([
        pd.DataFrame({'value': train_series, 'dataset': 'Train'}),
        pd.DataFrame({'value': valid_series, 'dataset': 'Validation'}),
        pd.DataFrame({'value': test_series, 'dataset': 'Test'})
    ])
    sns.boxplot(x='dataset', y='value', data=data, 
                palette=['blue', 'green', 'red'])
    plt.title(f'Boxplot Comparison of {col_name}', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel(col_name, fontsize=12)

    plt.tight_layout()
    plt.show()