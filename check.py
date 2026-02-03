import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import logging
from datetime import datetime
import os
import sys


# 设置中文字体
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass


# 配置日志系统
def setup_logging(log_dir="analysis_logs"):
    """配置日志系统并返回日志文件路径"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"white_noise_analysis_{timestamp}.log")

    if sys.stdout.encoding != 'UTF-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'UTF-8':
        sys.stderr.reconfigure(encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def check_white_noise(series, lags=37, alpha=0.05):
    """
    检查时序数据是否为白噪声
    返回: (检验结果DataFrame, 是否为白噪声的布尔值)
    """
    lb_test = acorr_ljungbox(series, lags=[lags], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    is_white_noise = p_value > alpha

    logging.info("=== Ljung–Box Q 检验结果 ===")
    logging.info(f"\n{lb_test.to_string()}")

    conclusion = (f"结论: p={p_value:.4f} > {alpha}, 接受原假设 → 序列近似白噪声"
                  if is_white_noise else
                  f"结论: p={p_value:.4f} ≤ {alpha}, 拒绝原假设 → 序列存在相关性")
    logging.info(conclusion)

    # 绘制并保存ACF图
    plt.figure(figsize=(7, 4))
    plot_acf(series, lags=lags)
    plt.title(f"自相关函数 (ACF) - {series.name}")

    if hasattr(series, 'name'):
        img_path = f"analysis_logs/acf_{series.name.replace('/', '_')}.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"ACF图已保存至: {img_path}")
    else:
        plt.show()

    return lb_test, is_white_noise


def analyze_agents(data):
    """分析agent级别的白噪声特性"""
    total_agents = data['agent_name'].nunique()
    logging.info(f"总共 {total_agents} 个agent")

    skipped_count = 0
    analyzed_count = 0
    white_noise_count = 0
    correlated_count = 0
    white_noise_agents = []
    correlated_agents = []

    for agent, group in data.groupby('agent_name'):
        group = group.sort_values('date_id').reset_index(drop=True)

        if len(group) < 37:
            skipped_count += 1
            logging.info(f"\n跳过Agent: {agent} (数据量不足: {len(group)} < 37)")
            continue

        analyzed_count += 1
        logging.info(f"\n\n===== 分析Agent: {agent} =====")
        logging.info(f"数据时间段: {group['date_id'].min()} 至 {group['date_id'].max()}")
        logging.info(f"有效数据点: {len(group)}")

        series = group['activity_group_amt']
        series.name = agent

        try:
            _, is_white_noise = check_white_noise(series)
            if is_white_noise:
                white_noise_count += 1
                white_noise_agents.append(agent)
            else:
                correlated_count += 1
                correlated_agents.append(agent)
        except Exception as e:
            logging.error(f"分析Agent {agent} 时出错: {str(e)}")
            continue

    return {
        'total_agents': total_agents,
        'analyzed_count': analyzed_count,
        'skipped_count': skipped_count,
        'white_noise_count': white_noise_count,
        'correlated_count': correlated_count,
        'white_noise_agents': white_noise_agents,
        'correlated_agents': correlated_agents
    }


def analyze_teams(data):
    """分析team级别的白噪声特性（保持与agent分析相同的逻辑）"""
    # 统计团队数量
    total_teams = data['team_id'].nunique()
    logging.info(f"总共 {total_teams} 个团队")

    # 初始化统计变量
    skipped_count = 0
    analyzed_count = 0
    white_noise_count = 0
    correlated_count = 0
    white_noise_teams = []
    correlated_teams = []

    # 按团队分组分析
    for team_id, group in data.groupby('team_id'):
        # 按日期排序
        group = group.sort_values('date_id').reset_index(drop=True)

        # 跳过数据量不足的团队
        if len(group) < 37:
            skipped_count += 1
            logging.info(f"\n跳过Team: {team_id} (数据量不足: {len(group)} < 37)")
            continue

        analyzed_count += 1
        logging.info(f"\n\n===== 分析Team: {team_id} =====")
        logging.info(f"数据时间段: {group['date_id'].min()} 至 {group['date_id'].max()}")
        logging.info(f"有效数据点: {len(group)}")
        logging.info(f"平均出勤人数: {group['on_duty_num'].mean():.1f}")

        # 准备时间序列数据
        series = group['activity_group_amt']
        series.name = f"Team_{team_id}"

        # 白噪声检验
        try:
            _, is_white_noise = check_white_noise(series)
            if is_white_noise:
                white_noise_count += 1
                white_noise_teams.append(team_id)
            else:
                correlated_count += 1
                correlated_teams.append(team_id)
        except Exception as e:
            logging.error(f"分析Team {team_id} 时出错: {str(e)}")
            continue

    # 返回结果字典
    return {
        'total_teams': total_teams,
        'analyzed_count': analyzed_count,
        'skipped_count': skipped_count,
        'white_noise_count': white_noise_count,
        'correlated_count': correlated_count,
        'white_noise_teams': white_noise_teams,
        'correlated_teams': correlated_teams
    }


def print_results(result, analysis_type="Agent"):
    """打印分析结果"""
    logging.info(f"\n分析完成统计 ({analysis_type}级别):")
    logging.info(f"总共{analysis_type}数量: {result[f'total_{analysis_type.lower()}s']}")
    logging.info(f"已分析{analysis_type}数量: {result['analyzed_count']}")
    logging.info(f"跳过{analysis_type}数量: {result['skipped_count']} (数据量不足)")
    logging.info(
        f"\n白噪声序列数量: {result['white_noise_count']} (占比: {result['white_noise_count'] / result['analyzed_count']:.2%})")
    logging.info(
        f"\n有相关性序列数量: {result['correlated_count']} (占比: {result['correlated_count'] / result['analyzed_count']:.2%})")

    logging.info(f"\n=== 白噪声{analysis_type}名单 ===")
    for item in result[f'white_noise_{analysis_type.lower()}s']:
        logging.info(item)

    logging.info(f"\n=== 有相关性{analysis_type}名单 ===")
    for item in result[f'correlated_{analysis_type.lower()}s']:
        logging.info(item)


def main():
    """主程序"""
    set_chinese_font()
    log_file = setup_logging()
    logging.info("开始白噪声分析...")

    # 加载数据（这里需要替换为实际的数据加载方式）
    data = pd.read_csv("./data/processed_202506160140_1.csv")
    data['date_id'] = pd.to_datetime(data['date_id'])
    logging.info(f"数据加载完成，共 {len(data)} 条记录")

    # 选择分析级别（Agent或Team）
    analysis_level = "Team"  # 可更改为"Agent"

    if analysis_level == "Agent":
        result = analyze_agents(data)
    else:
        result = analyze_teams(data)

    print_results(result, analysis_level)
    logging.info(f"\n详细日志已保存至: {log_file}")

    # 控制台输出最终统计
    print("\n=== 最终统计结果 ===")
    print(f"已分析{analysis_level}数量: {result['analyzed_count']}")
    print(
        f"白噪声序列数量: {result['white_noise_count']} (占比: {result['white_noise_count'] / result['analyzed_count']:.2%})")
    print(
        f"有相关性序列数量: {result['correlated_count']} (占比: {result['correlated_count'] / result['analyzed_count']:.2%})")


if __name__ == "__main__":
    main()