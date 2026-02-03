import os
from glob import glob

import pandas as pd


def process_data(data):
    # 识别需要处理的列（通常是字符串列）
    string_cols = ['agent_name', 'date_id', 'amt_mark_type', 'dx_group', 'group_type',
                   'city', 'sales_section', 'sales_office', 'team_id', 'on_duty_judgment']

    # 批量处理所有字符串列
    for col in string_cols:
        if col in data.columns:
            data[col] = data[col].str.strip('="')

    # 首先将date_id转换为datetime类型
    data['date_id'] = pd.to_datetime(data['date_id'])

    # 定义需要检查的列
    denominator_cols = ['to_seat_order', 'call_answer_15s_num', 'order_num']
    talk_dur_cols = ['predict_talk_dur', 'preview_talk_dur', 'cruise_talk_dur', 'sub_manual_talk_dur']
    numerator_cols = ['call_num', '15s_talk_dur', 'call_answer_15s_num']
    target_col = ['activity_group_amt']

    # 过滤条件：
    # 1. 分母列不能是 NaN 或 ≤ 0
    # 2. 通话时长不能是 NaN 或 < 0
    # 3. 分子列不能是 NaN
    # 4. 目标列不能是NaN 或 < 0
    valid_mask = (
            (data[denominator_cols].notna() & (data[denominator_cols] > 0)).all(axis=1) &  # 分母有效
            (data[talk_dur_cols].notna() & (data[talk_dur_cols] >= 0)).all(axis=1) &  # 通话时长有效
            (data[numerator_cols].notna()).all(axis=1) &  # 分子有效
            (data[target_col].notna() & (data[target_col] >= 0)).all(axis=1)  # 目标列NaN
    )

    # 应用过滤 - 这里添加.copy()创建独立副本
    filtered_data = data[valid_mask].copy()

    # 拨打比 = call_num / to_seat_order
    filtered_data['call_ratio'] = filtered_data['call_num'] / filtered_data['to_seat_order']

    # 15秒有效ATT = talk_dur_15s / call_answer_15s_num
    filtered_data['avg_15s_att'] = filtered_data['15s_talk_dur'] / filtered_data['call_answer_15s_num']

    # 总通话时长（小时）= 所有通话时长之和 / 3600
    filtered_data['avg_total_talk_dur_hr'] = filtered_data[talk_dur_cols].sum(axis=1) / 3600

    # 15秒有效接通比 = call_answer_15s_num / order_num
    filtered_data['call_15s_ans_ratio'] = filtered_data['call_answer_15s_num'] / filtered_data['order_num']

    # 查看过滤后的数据统计信息
    print(filtered_data[['call_ratio', 'avg_15s_att', 'call_15s_ans_ratio', 'avg_total_talk_dur_hr']].describe())

    # 过滤原始数据，并只保留指定列
    columns_to_keep = [
        'agent_name', 'date_id',  # 保留name和date_id用于后续分析
        'avg_15s_att',
        'call_ratio',
        'avg_total_talk_dur_hr',
        'call_15s_ans_ratio',
        'activity_group_amt'
    ]

    result_data = filtered_data[columns_to_keep].copy()  # 这里也添加.copy()

    # 日期特征 - 使用.loc进行赋值以避免警告
    result_data.loc[:, "day"] = result_data["date_id"].dt.day
    result_data.loc[:, "day_of_week"] = result_data["date_id"].dt.dayofweek

    return result_data


def main():
    results = []
    for i in range(3, 5):
        data_path = f'./data/zhuashou_{i}.csv'
        data = pd.read_csv(data_path, encoding='utf-8')
        results.append(process_data(data))

    result = pd.concat(results, ignore_index=True).sort_values(['agent_name', 'date_id'])

    # 筛选出满足最小天数的name
    name_counts = result.groupby('agent_name')['date_id'].nunique()
    valid_names = name_counts[name_counts >= 37].index

    result = result[result['agent_name'].isin(valid_names)]

    print(result['activity_group_amt'].describe())
    result.to_csv(f'./data/zhuashou_5_8.csv', index=False, encoding='utf-8')


def load_and_deduplicate(file_paths, out_conflict="conflicts.csv",
                         agent_col="agent_name", date_col="date_id",
                         target_col="activity_group_amt"):
    # 1. 读取所有文件
    dfs = [pd.read_csv(f) for f in file_paths]
    data = pd.concat(dfs, ignore_index=True)

    # 新增：如果指定了target_col，先排除target_col为0的行
    if target_col is not None and target_col in data.columns:
        original_count = len(data)
        data = data[data[target_col] != 0]
        filtered_count = len(data)
        print(f"已排除 {original_count - filtered_count} 条 {target_col}=0 的记录")

    # 2. 找出冲突行（同一个 key 出现多次）
    dup_mask = data.duplicated(subset=[agent_col, date_col], keep=False)
    conflicts = data[dup_mask].sort_values([agent_col, date_col])

    # 3. 保存冲突行
    if not conflicts.empty:
        conflicts.to_csv(out_conflict, index=False)
        print(f"发现 {len(conflicts)} 条冲突记录，已保存到 {out_conflict}")
    else:
        print("没有发现冲突记录")

    # 4. 去重（只保留第一次出现的）
    data = data.drop_duplicates(subset=[agent_col, date_col], keep="first")

    # 5. 排序
    data = data.sort_values(by=[agent_col, date_col]).reset_index(drop=True)

    return data


def process_team_data(data):
    """
    完全按照team_id分组合并数据后计算指标
    特点：
    1. 仅保留核心业务指标
    2. 彻底丢弃agent级别信息
    3. 严格的数据校验和过滤

    参数:
        data: 原始数据DataFrame

    返回:
        团队级别的核心指标DataFrame
    """
    # === 1. 数据预处理 ===
    # 处理字符串列的特殊字符
    string_cols = ['date_id', 'amt_mark_type', 'dx_group', 'group_type',
                   'city', 'sales_section', 'sales_office']

    for col in string_cols:
        if col in data.columns:
            data[col] = data[col].str.strip('="')

    # 日期转换
    data['date_id'] = pd.to_datetime(data['date_id'])

    # === 2. 定义所有数值列 ===
    numeric_cols = [
        'activity_group_amt', 'loan_m', 'begin_plan_agent_num',
        'manpower_oldtype_num', 'newcomer_activity_group_loan_amt',
        'manpower_newtype_num', 'mature_activity_group_loan_amt',
        'predict_talk_dur', 'preview_talk_dur', 'cruise_talk_dur',
        'sub_manual_talk_dur', 'predict_call_answer_num',
        'preview_call_answer_num', 'cruise_call_answer_num',
        'sub_manual_call_answer_num', 'call_num', 'to_seat_order',
        'on_duty_num', 'manual_call_num', 'call_answer_num',
        'call_answer_15s_num', 'talk_dur', '15s_talk_dur',
        'overtime_talk_dur', 'order_num', 'manual_call_answer_num'
    ]
    numeric_cols = [col for col in numeric_cols if col in data.columns]

    # === 3. 数据分组聚合 ===
    agg_dict = {
        # 团队信息列
        'city': 'first',
        'sales_section': 'first',
        'sales_office': 'first',

        # 数值列全部求和
        **{col: 'sum' for col in numeric_cols},

        # 分类列取众数
        'amt_mark_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'dx_group': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'group_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
        'on_duty_judgment': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
    }

    # 按team_id和date_id分组聚合
    team_data = data.groupby(['team_id', 'date_id']).agg(agg_dict).reset_index()

    # === 4. 数据清洗和异常值处理 ===
    # 定义关键指标列
    denominator_cols = ['to_seat_order', 'call_answer_15s_num', 'order_num']
    talk_dur_cols = ['predict_talk_dur', 'preview_talk_dur', 'cruise_talk_dur', 'sub_manual_talk_dur']
    numerator_cols = ['call_num', '15s_talk_dur', 'call_answer_15s_num']
    target_col = ['activity_group_amt']

    # 创建有效性掩码
    valid_mask = (
        # 基础非空检查
            team_data[numeric_cols].notna().all(axis=1) &

            # 业务逻辑检查
            (team_data[talk_dur_cols] >= 0).all(axis=1) &
            (team_data[denominator_cols] > 0).all(axis=1) &
            (team_data[target_col] >= 0).all(axis=1) &
            (team_data['on_duty_num'] > 0) &
            (team_data['call_num'] >= team_data['call_answer_num']) &
            (team_data['call_answer_num'] >= team_data['call_answer_15s_num'])
    )

    # 应用过滤
    clean_team_data = team_data[valid_mask].copy()

    # === 5. 计算核心业务指标 ===
    # 通话效率指标
    clean_team_data['call_ratio'] = clean_team_data['call_num'] / clean_team_data['to_seat_order']
    clean_team_data['avg_15s_att'] = clean_team_data['15s_talk_dur'] / clean_team_data['call_answer_15s_num']
    clean_team_data['avg_total_talk_dur_hr'] = clean_team_data[talk_dur_cols].sum(axis=1) / 3600
    clean_team_data['call_15s_ans_ratio'] = clean_team_data['call_answer_15s_num'] / clean_team_data['order_num']

    # === 6. 数据输出准备 ===
    # 添加日期特征
    clean_team_data['day'] = clean_team_data['date_id'].dt.day
    clean_team_data['day_of_week'] = clean_team_data['date_id'].dt.dayofweek
    clean_team_data['month'] = clean_team_data['date_id'].dt.month

    # 定义输出列（仅保留核心指标）
    output_columns = [
        # 团队标识信息
        'team_id', 'date_id',

        # 基础运营数据
        'on_duty_num', 'activity_group_amt',

        # 核心业务指标
        'call_ratio',
        'avg_15s_att',
        'call_15s_ans_ratio',
        'avg_total_talk_dur_hr',

        # 日期特征
        'day', 'day_of_week', 'month'
    ]

    # 打印处理报告
    print("=== 团队数据处理报告 ===")
    print(f"原始数据行数: {len(data)}")
    print(f"有效团队记录数: {len(clean_team_data)}")
    print("\n核心指标统计:")
    print(clean_team_data[['call_ratio', 'avg_15s_att', 'call_15s_ans_ratio']].describe().round(2))

    return clean_team_data[output_columns]


def process_data_file(input_file):
    """
    处理单个数据文件的完整流程
    1. 读取数据
    2. 处理数据
    3. 保存结果

    参数:
        input_file: 输入文件路径

    返回:
        输出文件路径
    """
    # 读取数据
    try:
        raw_data = pd.read_csv(input_file, encoding='utf-8')
        print(f"成功读取文件: {input_file}, 共 {len(raw_data)} 行数据")
    except Exception as e:
        print(f"读取文件失败: {input_file}")
        print(f"错误信息: {str(e)}")
        return None

    # 处理数据
    processed_data = process_team_data(raw_data)

    # 准备输出文件名
    file_dir = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    output_name = f"processed_{file_name}"
    output_path = os.path.join(file_dir, output_name)

    # 保存结果
    try:
        processed_data.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
        print(f"处理后的数据量: {len(processed_data)} 行")
        return output_path
    except Exception as e:
        print(f"保存文件失败: {output_path}")
        print(f"错误信息: {str(e)}")
        return None

if __name__ == '__main__':
    # main()
    # file_paths = glob("data/zhuashou_*.csv")
    # all_data = load_and_deduplicate(file_paths, out_conflict="conflicts.csv")
    # all_data["date_id"] = pd.to_datetime(all_data["date_id"])
    #
    # split_date = pd.to_datetime("2025-06-15")
    # train_df = all_data[all_data["date_id"] <= split_date]
    # test_df = all_data[all_data["date_id"] > split_date]
    #
    # print(all_data['activity_group_amt'].describe())
    # print(train_df['activity_group_amt'].describe())
    # print(test_df['activity_group_amt'].describe())
    #
    # all_data.to_csv(f'./data/zhuashou.csv', index=False, encoding='utf-8')
    #
    # train_df.to_csv(f'./data/zhuashou_20250301_20250615.csv', index=False, encoding='utf-8')
    # test_df.to_csv(f'./data/zhuashou_20250616_20250826.csv', index=False, encoding='utf-8')
    # 指定输入文件路径
    input_file = "./data/202506160140_1.csv"  # 请确保文件存在

    # 处理文件
    print("=== 开始处理数据 ===")
    result_file = process_data_file(input_file)

    if result_file:
        print("\n处理完成！结果文件:", result_file)
    else:
        print("\n处理过程中出现错误，请检查输入文件和错误信息")