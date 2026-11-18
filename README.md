# NILM 负荷分解模型复现与对比（Seq2Point / CNN / WaveNet / BGRU）

基于 PyTorch 的非侵入式负荷监测（NILM）实验框架，在 **UK-DALE House 1** 数据集上复现并对比多种深度学习负荷分解模型，提供统一配置入口的 Notebook 实验流程。

> 本项目由本科毕业设计整理而来，实验框架源自导师姜洁老师的 [fast-seq2point](https://github.com/jiejiang-jojo/fast-seq2point)，本人将其改造为可配置的多模型实验流程，并补充了训练工具、论文图表生成与结果汇总脚本。

## 项目特点

- **一个配置切换所有模型**：在 Notebook 第 1 节修改 `CONFIG['model']` 与 `model_params` 即可在 Seq2Point / CNN3 / CNN5 / CNN7 / WaveNet / BGRU / WaveNetBGRU 等模型间切换，无需改动训练代码
- **完整实验闭环**：数据生成 → 训练（含验证与最优模型保存）→ 推理 → 反归一化还原 → 预测曲线可视化 → 误差统计 → 结果汇总
- **规范的评估体系**：同时计算回归指标（MAE / RMSE / R²）与信号聚合误差 SAE，并支持二值化分类指标
- **可复现**：固定训练/验证/推理房屋划分与随机种子，每个模型-电器组合的训练日志与最优权重按迭代步数保存

## 支持的模型

| 模型 | 类型 | 说明 |
|---|---|---|
| Seq2Point | CNN | 序列到点的滑动窗口中点预测（基线） |
| CNN3 / CNN5 / CNN7 | CNN | 不同深度的卷积网络基线 |
| WaveNet | 膨胀因果卷积 | 残差 + 跳跃连接，感受野随层数指数增长 |
| BGRU | 双向 GRU | 双向时序建模，长窗口输入 |
| WaveNetBGRU | 混合 | 膨胀卷积与双向 GRU 组合（含加速版本） |
| LSTM | RNN | 原始 LSTM 实现 |

> CNN3 作为早期基线，因超参数与收敛表现较差未纳入论文对比，代码保留以供参考。

## 实验结果（UK-DALE House 1，四类电器）

以 MAE（W）为主要指标，SAE 为信号聚合误差：

| 模型 | 洗碗机 MAE / SAE | 电水壶 MAE / SAE | 微波炉 MAE / SAE | 洗衣机 MAE / SAE |
|---|---|---|---|---|
| CNN3 | 5.51 / 0.25 | 3.48 / 0.06 | 6.28 / 0.73 | 9.33 / 0.02 |
| WaveNet | 2.10 / 0.01 | 1.85 / 0.03 | 4.74 / 0.47 | 7.04 / 0.04 |
| BGRU | 1.57 / 0.02 | 1.80 / 0.02 | 5.56 / 0.61 | 5.54 / 0.09 |

结论：WaveNet 在多类电器上整体优于 CNN3 基线；BGRU 在洗碗机与洗衣机上 MAE 最低；微波炉因功率波动大、事件稀疏，三类模型误差均偏高。

推理耗时（测试集 1,298,298 条，RTX 5060 8GB）：CNN 约 20s、BGRU 约 55s、WaveNet 约 60s（GPU）。

## 项目结构

```
.
├── configs/example_config.json    # 训练配置模板（模型/窗口/学习率等）
├── notebooks/run_all_models.ipynb # ⭐ 统一实验入口（可配置模型）
├── src/
│   ├── models.py                  # 全部模型定义
│   ├── data_generator.py          # 原始数据生成器（训练/测试）
│   ├── data_generators.py         # 改造后的生成器
│   ├── utilities.py               # 日志、指标、参数保存等通用工具
│   ├── training_utils.py          # 训练/验证循环封装
│   ├── main_pytorch.py            # 命令行训练入口
│   ├── pack_csv_to_hdf5.py        # CSV → HDF5 数据打包
│   └── data_stats.py              # 数据集统计
├── scripts/
│   ├── generate_paper_figures.py  # 论文图表批量生成
│   └── generate_summary.py        # 实验结果汇总
└── requirements.txt
```

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

Python 3.10+，PyTorch 2.x；GPU 可选（CUDA 可用时自动加速）。

### 2. 准备数据

下载 [UK-DALE 数据集](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyDisConsumption/2014)（House 1，采样率 1/8 Hz），将 `house_1` 的通道 CSV 整理为 `house1.csv`，再打包为 HDF5：

```bash
python src/pack_csv_to_hdf5.py --csv_path ./house1.csv --h5_path ./data.h5
```

> 数据文件体积较大（`data.h5` 约 240MB），未随仓库提供。实验选取 House 1 连续两个月的数据，按 70% / 15% / 15% 划分训练、验证与测试集。

### 3. 运行实验

打开 `notebooks/run_all_models.ipynb`，在第 1 节修改配置：

```python
CONFIG = {
    'model': 'WaveNet',          # Seq2Point / CNN3 / CNN5 / CNN7 / WaveNet / BGRU ...
    'model_params': {...},        # 各模型专属参数
    'target_device': 'washingmachine',  # dishwasher / kettle / microwave / washingmachine
    'train_house_list': ['house11'],
    'max_iterations': 300000,
    ...
}
```

依次运行即可获得训练曲线、最优模型、预测对比图与误差统计。

### 4. 命令行训练（可选）

```bash
python src/main_pytorch.py --model_type WaveNet --target_device washingmachine \
    --hdf5_path ./data.h5 --workspace ./workspace
```

## 我的贡献

相对上游 [fast-seq2point](https://github.com/jiejiang-jojo/fast-seq2point)：

1. 将分散的命令行实验流程重构为**单个可配置 Notebook**，支持模型、窗口长度、训练轮次、目标电器、房屋划分的参数化切换
2. 基于 UK-DALE House 1 的两个月数据重新构建训练/验证/测试流程，统一了归一化与反归一化的处理链路
3. 新增 `training_utils.py`（训练/验证循环）、`generate_paper_figures.py`（论文图表批量生成）、`generate_summary.py`（多实验指标汇总）
4. 完成 WaveNet、BGRU、WaveNetBGRU 等模型在本实验设定下的训练与调参，并与 CNN3 基线做系统对比

## 致谢与许可

- 实验框架基于姜洁老师开源的 [fast-seq2point](https://github.com/jiejiang-jojo/fast-seq2point)（仓库未声明开源许可证，本项目在其基础上整理，已获作者许可，仅用于学术研究与学习交流）
- 数据集：Jack Kelly et al., *UK-DALE: A dataset recording UK Domestic Appliance-Level Electricity demand and whole-house demand*
- 本人新增的代码（Notebook 实验流程、训练工具、图表与汇总脚本）以 MIT 协议开源，见 [LICENSE](LICENSE)

## 相关项目

配套的可视化分析平台：[nilm-visual-platform](https://github.com/<your-username>/nilm-visual-platform)（FastAPI + React，支持模型预测、指标计算与分功率区间误差分析）
