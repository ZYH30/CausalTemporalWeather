import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path


def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    df["date_id"] = pd.to_datetime(df["date_id"])
    return df


def create_plot_directories(base_dir, sample_num):
    """为每个样本创建图表目录结构"""
    sample_dir = os.path.join(base_dir, f"sample_{sample_num}")
    dirs = {
        'main': sample_dir,
        'activity_grid': os.path.join(sample_dir, "activity_grid"),
        'individual_plots': os.path.join(sample_dir, "individual_plots"),
        'statistics': os.path.join(sample_dir, "statistics"),
        'comparison_charts': os.path.join(sample_dir, "statistics", "comparison_charts")
    }

    # 创建所有需要的目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def plot_agent_activity(df, dirs, sample_num):
    """
    绘制坐席活动时序图
    Args:
        df: 包含抽样数据的DataFrame
        dirs: 目录路径字典
        sample_num: 当前样本编号
    """
    # 1. 联合图表（FacetGrid）
    g = sns.FacetGrid(df, col="agent_name", col_wrap=4, height=3, sharey=False)
    g.map_dataframe(sns.lineplot, x="date_id", y="activity_group_amt", marker="o")
    g.set_axis_labels("Date", "Activity Volume")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Activity Volume Time Series (Sample {sample_num})", fontsize=14)
    plt.savefig(os.path.join(dirs['activity_grid'], 'activity_grid.png'), dpi=300)
    plt.close()

    # 2. 单个坐席图表
    for agent, group in df.groupby("agent_name"):
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=group, x="date_id", y="activity_group_amt", marker="o")
        mean_val = group["activity_group_amt"].mean()
        plt.axhline(mean_val, color="gray", linestyle="--", alpha=0.6, label=f"Mean={mean_val:.1f}")
        plt.title(f"Agent {agent} (Sample {sample_num})")
        plt.xlabel("Date")
        plt.ylabel("Activity Volume")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['individual_plots'], f"{agent}.png"), dpi=300)
        plt.close()


def calculate_agent_stats(df, dirs, sample_num):
    """
    计算并可视化坐席统计数据
    Args:
        df: 包含抽样数据的DataFrame
        dirs: 目录路径字典
        sample_num: 当前样本编号
    Returns:
        统计结果DataFrame
    """
    # 计算基础统计
    stats = df.groupby('agent_name')['activity_group_amt'].agg(
        ['count', 'min', 'max', 'mean', 'median']
    ).reset_index()
    stats.columns = ['Agent', 'Records', 'Min', 'Max', 'Mean', 'Median']

    # 保存统计结果
    stats.to_csv(os.path.join(dirs['statistics'], 'summary_stats.csv'), index=False)

    # 绘制各指标对比图
    metrics = ['Min', 'Max', 'Mean', 'Median']
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.barplot(data=stats, x='Agent', y=metric)
        plt.title(f'{metric} Comparison')
        plt.xlabel('')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    plt.suptitle(f"Agent Metrics Comparison (Sample {sample_num})")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['comparison_charts'], 'metrics_comparison.png'), dpi=300)
    plt.close()

    # 单独绘制记录数图表
    plt.figure(figsize=(12, 6))
    sns.barplot(data=stats, x='Agent', y='Records')
    plt.title(f'Record Count per Agent (Sample {sample_num})')
    plt.xlabel('Agent')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['comparison_charts'], 'record_count.png'), dpi=300)
    plt.close()

    return stats


def run_sampling_analysis(file_path, sample_size=20, num_samples=10, output_base="sampling_results"):
    """
    执行多次抽样分析
    Args:
        file_path: 数据文件路径
        sample_size: 每次抽样的坐席数量
        num_samples: 抽样次数
        output_base: 输出基础目录
    """
    full_df = load_data(file_path)
    all_agents = full_df['agent_name'].unique()

    if sample_size >= len(all_agents):
        print(f"注意：抽样数量{sample_size}≥总坐席数{len(all_agents)}，将使用全部数据")
        sample_size = None

    print(f"\n开始执行{num_samples}次抽样分析，每次抽样{sample_size or '全部'}个坐席...")

    for i in range(1, num_samples + 1):
        # 1. 创建目录结构
        dirs = create_plot_directories(output_base, i)

        # 2. 随机抽样
        if sample_size:
            random.seed(i)  # 使用样本编号作为随机种子
            sampled_agents = random.sample(list(all_agents), sample_size)
            sample_df = full_df[full_df['agent_name'].isin(sampled_agents)]
            print(f"样本{i}: 已抽样坐席 {len(sampled_agents)}个")
        else:
            sample_df = full_df.copy()
            print(f"样本{i}: 使用全部坐席 {len(all_agents)}个")

        # 3. 保存抽样信息
        with open(os.path.join(dirs['main'], 'sampling_info.txt'), 'w') as f:
            if sample_size:
                f.write(f"Sampled {sample_size} agents:\n")
                f.write("\n".join(sampled_agents))
            else:
                f.write("Used all available agents")

        # 4. 执行分析
        plot_agent_activity(sample_df, dirs, i)
        stats_df = calculate_agent_stats(sample_df, dirs, i)

        # 5. 保存抽样数据
        sample_df.to_csv(os.path.join(dirs['main'], 'sampled_data.csv'), index=False)

    print(f"\n分析完成！所有结果已保存到 {output_base}/ 目录")


if __name__ == "__main__":
    # 配置参数
    DATA_FILE = "./data/zhuashou.csv"  # 数据文件路径
    SAMPLE_SIZE = 20  # 每次抽样的坐席数量
    NUM_SAMPLES = 10  # 抽样次数
    OUTPUT_DIR = "multi_sample_results"  # 输出目录

    # 执行分析
    run_sampling_analysis(
        file_path=DATA_FILE,
        sample_size=SAMPLE_SIZE,
        num_samples=NUM_SAMPLES,
        output_base=OUTPUT_DIR
    )