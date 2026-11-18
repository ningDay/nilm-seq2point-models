"""
生成论文级汇总图表
用于NILM深度学习模型性能对比
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']

# 实验数据汇总
EXPERIMENTS = {
    'Seq2Point': {
        'dishwasher': {'MAE': 10.05, 'RMSE': 124.96, 'SAE': 0.11, 'best_iter': 283000},
        'kettle': {'MAE': 6.84, 'RMSE': 108.22, 'SAE': 0.39, 'best_iter': 271000},
        'microwave': {'MAE': 6.90, 'RMSE': 100.90, 'SAE': 0.86, 'best_iter': 282000},
        'washingmachine': {'MAE': 9.71, 'RMSE': 86.94, 'SAE': 0.21, 'best_iter': 267000},
        'seq_len': 15
    },
    'CNN3': {
        'dishwasher': {'MAE': 7.13, 'RMSE': 101.21, 'SAE': 0.31, 'best_iter': 298000},
        'kettle': {'MAE': 3.80, 'RMSE': 75.66, 'SAE': 0.11, 'best_iter': 289000},
        'microwave': {'MAE': 6.17, 'RMSE': 92.50, 'SAE': 0.68, 'best_iter': 287000},
        'washingmachine': {'MAE': 9.48, 'RMSE': 81.11, 'SAE': 0.17, 'best_iter': 292000},
        'seq_len': 41
    },
    'WaveNet': {
        'dishwasher': {'MAE': 2.59, 'RMSE': 36.08, 'SAE': 0.10, 'best_iter': 266000},
        'kettle': {'MAE': 1.68, 'RMSE': 52.88, 'SAE': 0.02, 'best_iter': 99000},
        'microwave': {'MAE': 5.26, 'RMSE': 84.66, 'SAE': 0.57, 'best_iter': 182000},
        'washingmachine': {'MAE': 5.67, 'RMSE': 52.66, 'SAE': 2.55, 'best_iter': 291000},
        'seq_len': 127
    },
    'BGRU': {
        'dishwasher': {'MAE': 1.57, 'RMSE': 38.57, 'SAE': 0.02, 'best_iter': 261000},
        'kettle': {'MAE': 1.80, 'RMSE': 59.10, 'SAE': 0.02, 'best_iter': 189000},
        'microwave': {'MAE': 5.56, 'RMSE': 88.78, 'SAE': 0.61, 'best_iter': 293000},
        'washingmachine': {'MAE': 5.54, 'RMSE': 61.55, 'SAE': 0.09, 'best_iter': 257000},
        'seq_len': 511
    }
}

# 模型参数量（估计值）
PARAMS = {
    'Seq2Point': 30000,
    'CNN3': 150000,
    'WaveNet': 200000,
    'BGRU': 500000
}

DEVICES = ['dishwasher', 'kettle', 'microwave', 'washingmachine']
MODELS = ['Seq2Point', 'CNN3', 'WaveNet', 'BGRU']

# 输出目录
OUTPUT_DIR = r'd:\GraduationDesign\ReplicateTest\fast-seq2point\outputs\figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色配置
COLORS = {
    'Seq2Point': '#1f77b4',
    'CNN3': '#ff7f0e',
    'WaveNet': '#2ca02c',
    'BGRU': '#d62728'
}

def generate_summary_table():
    """生成性能对比汇总表"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    for model in MODELS:
        for device in DEVICES:
            data = EXPERIMENTS[model][device]
            table_data.append([
                model,
                device,
                f"{data['MAE']:.2f}",
                f"{data['RMSE']:.2f}",
                f"{data['SAE']:.2f}",
                f"{data['best_iter']//1000}K"
            ])
    
    # 表格列名
    col_labels = ['模型', '电器', 'MAE (W)', 'RMSE (W)', 'SAE (%)', '最优迭代']
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.18, 0.14, 0.14, 0.14, 0.14]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if (i - 1) // len(DEVICES) % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('表4-1 各模型在测试集上的性能对比', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_table.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_table.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/summary_table.png")

def generate_mae_comparison():
    """生成MAE对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    device_names = ['洗碗机 (Dishwasher)', '电水壶 (Kettle)', '微波炉 (Microwave)', '洗衣机 (Washingmachine)']
    
    for idx, device in enumerate(DEVICES):
        ax = axes[idx]
        models = list(MODELS)
        mae_values = [EXPERIMENTS[m][device]['MAE'] for m in models]
        colors = [COLORS[m] for m in models]
        
        bars = ax.bar(models, mae_values, color=colors, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, mae_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('MAE (W)', fontsize=11)
        ax.set_title(device_names[idx], fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(mae_values) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('图4-1 各模型在不同电器上的MAE对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mae_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mae_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/mae_comparison.png")

def generate_model_comparison_heatmap():
    """生成模型-电器性能热力图"""
    # 提取MAE矩阵
    mae_matrix = np.array([[EXPERIMENTS[m][d]['MAE'] for d in DEVICES] for m in MODELS])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # 添加数值标签
    for i in range(len(MODELS)):
        for j in range(len(DEVICES)):
            text = ax.text(j, i, f'{mae_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    
    # 设置轴标签
    device_labels = ['Dishwasher', 'Kettle', 'Microwave', 'Washingmachine']
    ax.set_xticks(np.arange(len(DEVICES)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels(device_labels, fontsize=10)
    ax.set_yticklabels(MODELS, fontsize=10)
    
    ax.set_xlabel('电器 (Appliance)', fontsize=12)
    ax.set_ylabel('模型 (Model)', fontsize=12)
    ax.set_title('图4-2 模型-电器MAE性能热力图 (越绿越好)', fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('MAE (W)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_heatmap.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/model_heatmap.png")

def generate_radar_chart():
    """生成雷达图比较各模型在不同指标上的表现"""
    # 计算各模型在各电器上的平均表现（反向，归一化到0-1，越高越好）
    metrics = ['MAE', 'RMSE', 'SAE']
    
    # 各模型平均指标
    model_avg = {}
    for model in MODELS:
        mae_vals = [EXPERIMENTS[model][d]['MAE'] for d in DEVICES]
        rmse_vals = [EXPERIMENTS[model][d]['RMSE'] for d in DEVICES]
        sae_vals = [EXPERIMENTS[model][d]['SAE'] for d in DEVICES]
        model_avg[model] = {
            'MAE': np.mean(mae_vals),
            'RMSE': np.mean(rmse_vals),
            'SAE': np.mean(sae_vals)
        }
    
    # 归一化（转换为分数，越低指标越好，所以用1-归一化值）
    all_mae = [model_avg[m]['MAE'] for m in MODELS]
    all_rmse = [model_avg[m]['RMSE'] for m in MODELS]
    all_sae = [model_avg[m]['SAE'] for m in MODELS]
    
    # 计算分数（转换为0-100分数，越高越好）
    def normalize_inverse(values):
        max_val = max(values)
        min_val = min(values)
        if max_val == min_val:
            return [50] * len(values)
        return [100 * (max_val - v) / (max_val - min_val) for v in values]
    
    mae_scores = normalize_inverse(all_mae)
    rmse_scores = normalize_inverse(all_rmse)
    sae_scores = normalize_inverse(all_sae)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, model in enumerate(MODELS):
        values = [mae_scores[i], rmse_scores[i], sae_scores[i]]
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=COLORS[model])
        ax.fill(angles, values, alpha=0.15, color=COLORS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('图4-3 各模型综合性能雷达图\n(分数越高表示该指标越好)', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/radar_chart.png")

def generate_complexity_performance():
    """生成模型复杂度与性能权衡图"""
    # 各模型参数量和平均MAE
    params = [PARAMS[m] / 1000 for m in MODELS]  # 转换为K参数
    avg_mae = [np.mean([EXPERIMENTS[m][d]['MAE'] for d in DEVICES]) for m in MODELS]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # MAE柱状图
    bars = ax1.bar(MODELS, avg_mae, color=[COLORS[m] for m in MODELS], 
                   edgecolor='black', linewidth=1, alpha=0.7, label='Avg MAE')
    ax1.set_ylabel('平均 MAE (W)', fontsize=12, color='black')
    ax1.set_xlabel('模型 (Model)', fontsize=12)
    ax1.set_ylim(0, max(avg_mae) * 1.3)
    
    # 添加MAE数值标签
    for bar, val in zip(bars, avg_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 参数量折线图（使用第二个Y轴）
    ax2 = ax1.twinx()
    line = ax2.plot(MODELS, params, 'ko-', linewidth=2, markersize=10, label='参数量')
    ax2.set_ylabel('参数量 (K)', fontsize=12, color='black')
    ax2.set_ylim(0, max(params) * 1.3)
    
    # 添加参数量数值标签
    for i, (m, p) in enumerate(zip(MODELS, params)):
        ax2.annotate(f'{int(p)}K', (i, p), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title('图4-4 模型复杂度与性能权衡', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'complexity_performance.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'complexity_performance.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/complexity_performance.png")

def generate_relative_improvement():
    """生成相对性能提升对比图（以Seq2Point为基准）"""
    baseline = 'Seq2Point'
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, device in enumerate(DEVICES):
        ax = axes[idx]
        
        baseline_mae = EXPERIMENTS[baseline][device]['MAE']
        models = MODELS[1:]  # 排除基准模型
        improvements = []
        
        for model in models:
            current_mae = EXPERIMENTS[model][device]['MAE']
            improvement = (baseline_mae - current_mae) / baseline_mae * 100
            improvements.append(improvement)
        
        colors = [COLORS[m] for m in models]
        bars = ax.bar(models, improvements, color=colors, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 1 if bar.get_height() >= 0 else bar.get_height() - 3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        device_names = ['Dishwasher', 'Kettle', 'Microwave', 'Washingmachine']
        ax.set_ylabel('相对MAE降低 (%)', fontsize=10)
        ax.set_title(f'{device_names[idx]}', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
    
    plt.suptitle('图4-5 各模型相对于Seq2Point的MAE降低率 (越高越好)', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'relative_improvement.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'relative_improvement.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/relative_improvement.png")

def generate_boxplot_comparison():
    """生成各模型跨电器的MAE箱线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    colors_list = []
    
    for model in MODELS:
        mae_vals = [EXPERIMENTS[model][d]['MAE'] for d in DEVICES]
        data.append(mae_vals)
        labels.append(model)
        colors_list.append(COLORS[model])
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # 设置箱线图颜色
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 添加数据点
    for i, (d, color) in enumerate(zip(data, colors_list)):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color=color, edgecolors='black', linewidth=0.5, s=50)
    
    ax.set_ylabel('MAE (W)', fontsize=12)
    ax.set_xlabel('模型 (Model)', fontsize=12)
    ax.set_title('图4-6 各模型在不同电器上的MAE分布箱线图', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {OUTPUT_DIR}/boxplot_comparison.png")

def generate_all_charts():
    """生成所有图表"""
    print("=" * 50)
    print("开始生成论文图表...")
    print("=" * 50)
    
    generate_summary_table()
    generate_mae_comparison()
    generate_model_comparison_heatmap()
    generate_radar_chart()
    generate_complexity_performance()
    generate_relative_improvement()
    generate_boxplot_comparison()
    
    print("=" * 50)
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    generate_all_charts()
