import pandas as pd
import numpy as np
from collections import defaultdict
import os


def find_samples_similar_in_all_features(df, features, threshold=0.05):
    """
    找出在所有指定特征上都相似的样本组

    参数:
        df: 包含数据的DataFrame
        features: 需要同时满足相似的特征列表
        threshold: 相似度阈值（默认5%）

    返回:
        列表：每个元素是一个相似样本组的DataFrame
    """
    # 按第一个特征排序作为初始分组依据
    df = df.sort_values(by=features[0]).reset_index(drop=True)

    groups = []
    current_group = []

    for idx, row in df.iterrows():
        if not current_group:
            current_group.append(idx)
            continue

        # 检查是否与当前组在所有特征上都相似
        is_similar = True
        for feat in features:
            group_avg = df.loc[current_group, feat].mean()
            current_val = row[feat]
            diff = abs(current_val - group_avg) / group_avg
            if diff > threshold:
                is_similar = False
                break

        if is_similar:
            current_group.append(idx)
        else:
            # 保存当前组并开始新组
            if len(current_group) > 1:
                groups.append(df.loc[current_group].copy())
            current_group = [idx]

    # 添加最后一组
    if len(current_group) > 1:
        groups.append(df.loc[current_group].copy())

    return groups


def analyze_multi_feature_similarity(input_file, output_dir, features, threshold=0.05):
    """
    分析在所有指定特征上相似的样本组

    参数:
        input_file: 输入数据路径
        output_dir: 输出目录
        features: 需要同时分析的特征列表
        threshold: 相似度阈值
    """
    # 读取数据
    df = pd.read_csv(input_file)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 找出相似样本组
    similar_groups = find_samples_similar_in_all_features(df, features, threshold)

    # 准备结果DataFrame
    results = []
    for i, group_df in enumerate(similar_groups, 1):
        # 计算组统计信息
        group_stats = {
            'group_id': i,
            'sample_count': len(group_df),
            'agents': ', '.join(group_df['agent_name'].unique()),
            'date_range': f"{group_df['date_id'].min()} to {group_df['date_id'].max()}"
        }

        # 添加每个特征的平均值
        for feat in features:
            group_stats[f'{feat}_avg'] = group_df[feat].mean()
            group_stats[f'{feat}_range'] = f"{group_df[feat].min():.2f}-{group_df[feat].max():.2f}"

        results.append(group_stats)

        # 保存该组的详细数据
        group_df.to_csv(os.path.join(output_dir, f"group_{i}_details.csv"), index=False)

    # 保存汇总结果
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(os.path.join(output_dir, "similar_groups_summary.csv"), index=False)

        print(f"找到 {len(results)} 组相似样本，结果已保存到 {output_dir}")
        print("\n相似样本组统计摘要:")
        print(summary_df[['group_id', 'sample_count', 'agents', 'date_range'] +
                         [f'{feat}_avg' for feat in features]])
    else:
        print("未找到满足条件的相似样本组")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_file = "./data/zhuashou.csv"
    output_dir = "./output/multi_feature_similarity"
    features_to_analyze = ['avg_15s_att', 'call_ratio', 'avg_total_talk_dur_hr', 'call_15s_ans_ratio']
    similarity_threshold = 0.05  # 5%

    # 执行分析
    analyze_multi_feature_similarity(
        input_file=input_file,
        output_dir=output_dir,
        features=features_to_analyze,
        threshold=similarity_threshold
    )