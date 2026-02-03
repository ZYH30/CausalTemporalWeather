# CausalTemporalWeather: 基于对抗因果学习的时间序列预测框架

## 1. 项目简介

**CausalTemporalWeather** 是一个集成了**因果推断 (Causal Inference)** 与**深度时序建模 (Deep Temporal Modeling)** 的实验性预测框架。该项目旨在解决传统时序模型（如 LSTM, Transformer）在处理包含协变量干扰和动态环境下的鲁棒性问题。

本项目核心引入了**对抗因果机制 (Adversarial Causal Learning)**，通过特征塔 (Feature Tower) 与目标塔 (Target Tower) 的解耦建模，结合对抗损失函数，力求在复杂的金融（收益率预测）或 气象（温度/湿度预测）任务中，识别并提取更具泛化能力的因果特征。

### 核心亮点

* **因果特征解耦**：利用双塔架构，将协变量的特征表征与目标预测逻辑分离。
* **对抗鲁棒性 (Adversarial Training)**：通过对抗性训练策略（Adversarial Loss），抑制特征中的虚假相关性。
* **多模型生态**：原生支持 `LSTMCausalAd`（因果对抗 LSTM）、`TimeMixer`、`Transformer` 等多种尖端时序模型。
* **维度自动校正**：具备运行时数据维度推断逻辑，能自动处理因特征工程或缺失值剔除导致的输入维度变化。

---

## 2. 算法架构

本项目的核心因果假设遵循**序贯可忽略性 (Sequential Ignorability)**。模型通过以下逻辑构建预测器：

### 核心组件说明

* **Feature Tower (特征塔)**：提取历史与未来协变量的隐藏层表征。
* **Target Tower (目标塔)**：引入自回归反馈，通过 Teacher Forcing (训练期) 或 自回归迭代 (推理期) 捕捉目标变量的内在惯性。
* **RevIN (可逆实例归一化)**：针对非平稳时序设计的标准化层，有效缓解分布偏移问题。

---

## 3. 文件结构

```bash
CausalTemporalWeather/
├── main.py                # 主程序入口：数据加载、维度校正、训练与评估循环
├── config.py              # 参数配置中心：定义超参数、模型结构及特征列表
├── train.py               # 训练逻辑：包含对抗训练 (Adversarial) 与常规训练逻辑
├── evaluate.py            # 评估引擎：计算 RMSE, MSE, MAE 及 GKX-R² 指标
├── loss.py                # 损失函数：自定义对抗损失 AdversarialLoss
├── util.py                # 工具库：数据预处理、Batch 生成器、随机种子固定
├── models/                # 模型定义
│   ├── base.py            # 基础组件 (RevIN, Attention, Encoder/Decoder)
│   ├── lstm_attn_cau.py   # 因果对抗 LSTM 模型实现
│   └── time_mixer_adapter.py # TimeMixer 适配器
└── best_params_30.yaml    # 预设的最佳超参数配置

```

---

## 4. 快速开始

### 安装依赖

```bash
pip install torch polars pandas joblib pyyaml matplotlib scikit-learn

```

### 运行实验

本项目支持通过命令行参数或配置文件快速切换实验任务：

**1. 默认因果对抗训练:**

```bash
python main.py \
  --model LSTMCausalAd \
  --data_path ./data/weather.csv \
  --target OT \
  --past_features wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K \
  --forward_features month year \
  --sequence_length 96 \
  --step_forward 96 \
  --batch_size 1024 \
  --lr 0.01 \
  --epochs 50 \
  --fix_seed True

```

**2. 使用 Shell 脚本运行 (如运行普通 LSTM 比较):**

```bash
bash runMain.sh  

```

**3. 执行超参数搜索:**

```bash
bash runSearch.sh

```

---

## 5. 指标评估

模型评估阶段不仅计算常规的误差指标，还特别引入了金融学术界常用的 **GKX (2020) Out-of-Sample R²**，以衡量模型相对于历史均值预测的超额解释能力。

---

## 6. 参数配置

主要超参数可在 `config.py` 或 `.yaml` 文件中调整：

| 参数名 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `LSTMCausalAd` | 选择模型架构 (TimeMixer, LSTM 等) |
| `adv_weight` | `0.001` | 对抗损失权重，控制解耦强度 |
| `sequence_length` | `96` | 历史输入序列长度 |
| `step_forward` | `96` | 预测步长 |
| `share_outNet` | `True` | Encoder/Decoder 是否共享输出层 |

---

## 7. 贡献指南

欢迎针对因果结构识别、双重稳健估计 (Doubly Robust Estimation) 以及 Transformer 变体集成方面的 PR。请确保提交的代码通过了 `check.py` 中的基础测试。
