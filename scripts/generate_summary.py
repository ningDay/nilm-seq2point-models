"""
整理现有实验图表，生成论文汇总
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = r'd:\GraduationDesign\ReplicateTest\fast-seq2point\outputs\figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验数据
EXPERIMENTS = {
    'Seq2Point': {
        'dishwasher': {'MAE': 10.05, 'RMSE': 124.96, 'SAE': 0.11, 'best_iter': 283000},
        'kettle': {'MAE': 6.84, 'RMSE': 108.22, 'SAE': 0.39, 'best_iter': 271000},
        'microwave': {'MAE': 6.90, 'RMSE': 100.90, 'SAE': 0.86, 'best_iter': 282000},
        'washingmachine': {'MAE': 9.71, 'RMSE': 86.94, 'SAE': 0.21, 'best_iter': 267000}
    },
    'CNN3': {
        'dishwasher': {'MAE': 7.13, 'RMSE': 101.21, 'SAE': 0.31, 'best_iter': 298000},
        'kettle': {'MAE': 3.80, 'RMSE': 75.66, 'SAE': 0.11, 'best_iter': 289000},
        'microwave': {'MAE': 6.17, 'RMSE': 92.50, 'SAE': 0.68, 'best_iter': 287000},
        'washingmachine': {'MAE': 9.48, 'RMSE': 81.11, 'SAE': 0.17, 'best_iter': 292000}
    },
    'WaveNet': {
        'dishwasher': {'MAE': 2.59, 'RMSE': 36.08, 'SAE': 0.10, 'best_iter': 266000},
        'kettle': {'MAE': 1.68, 'RMSE': 52.88, 'SAE': 0.02, 'best_iter': 99000},
        'microwave': {'MAE': 5.26, 'RMSE': 84.66, 'SAE': 0.57, 'best_iter': 182000},
        'washingmachine': {'MAE': 5.67, 'RMSE': 52.66, 'SAE': 2.55, 'best_iter': 291000}
    },
    'BGRU': {
        'dishwasher': {'MAE': 1.57, 'RMSE': 38.57, 'SAE': 0.02, 'best_iter': 261000},
        'kettle': {'MAE': 1.80, 'RMSE': 59.10, 'SAE': 0.02, 'best_iter': 189000},
        'microwave': {'MAE': 5.56, 'RMSE': 88.78, 'SAE': 0.61, 'best_iter': 293000},
        'washingmachine': {'MAE': 5.54, 'RMSE': 61.55, 'SAE': 0.09, 'best_iter': 257000}
    }
}

MODELS = ['Seq2Point', 'CNN3', 'WaveNet', 'BGRU']
DEVICES = ['dishwasher', 'kettle', 'microwave', 'washingmachine']
COLORS = {'Seq2Point': '#1f77b4', 'CNN3': '#ff7f0e', 'WaveNet': '#2ca02c', 'BGRU': '#d62728'}

def create_summary_table():
    """生成性能对比汇总表"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 表头
    header_color = '#4472C4'
    row_colors = ['#D6DCE4', '#FFFFFF']
    
    # 创建表格数据
    cell_text = []
    for i, model in enumerate(MODELS):
        for j, device in enumerate(DEVICES):
            d = EXPERIMENTS[model][device]
            cell_text.append([
                model, device,
                f"{d['MAE']:.2f}",
                f"{d['RMSE']:.2f}",
                f"{d['SAE']:.2f}",
                f"{d['best_iter']//1000}K"
            ])
    
    col_labels = ['Model', 'Appliance', 'MAE (W)', 'RMSE (W)', 'SAE (%)', 'Best Iter']
    
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.14, 0.16, 0.12, 0.12, 0.12, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # 设置表头
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # 设置行颜色
    for i in range(1, len(cell_text) + 1):
        color_idx = (i - 1) // len(DEVICES) % 2
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(row_colors[color_idx])
    
    plt.title('Table 1: Performance Comparison on Test Set', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_summary_table.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, '01_summary_table.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 01_summary_table.png")

def create_mae_comparison():
    """生成MAE对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    device_names = ['Dishwasher', 'Kettle', 'Microwave', 'Washingmachine']
    
    for idx, device in enumerate(DEVICES):
        ax = axes[idx]
        mae_values = [EXPERIMENTS[m][device]['MAE'] for m in MODELS]
        colors = [COLORS[m] for m in MODELS]
        
        bars = ax.bar(MODELS, mae_values, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, mae_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('MAE (W)', fontsize=11)
        ax.set_title(device_names[idx], fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(mae_values) * 1.2)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 1: MAE Comparison Across Models and Appliances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_mae_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, '02_mae_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 02_mae_comparison.png")

def create_heatmap():
    """生成热力图"""
    mae_matrix = np.array([[EXPERIMENTS[m][d]['MAE'] for d in DEVICES] for m in MODELS])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
    
    for i in range(len(MODELS)):
        for j in range(len(DEVICES)):
            ax.text(j, i, f'{mae_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    
    ax.set_xticks(np.arange(len(DEVICES)))
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_xticklabels([d.capitalize() for d in DEVICES], fontsize=10)
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_xlabel('Appliance', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Figure 2: MAE Heatmap (Lower is Better)', fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='MAE (W)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_mae_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_mae_heatmap.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 03_mae_heatmap.png")

def create_complexity_chart():
    """模型复杂度与性能权衡"""
    params = {'Seq2Point': 30, 'CNN3': 150, 'WaveNet': 200, 'BGRU': 500}  # K
    avg_mae = [np.mean([EXPERIMENTS[m][d]['MAE'] for d in DEVICES]) for m in MODELS]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bars = ax1.bar(MODELS, avg_mae, color=[COLORS[m] for m in MODELS], 
                   edgecolor='black', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Average MAE (W)', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylim(0, max(avg_mae) * 1.3)
    
    for bar, val in zip(bars, avg_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2 = ax1.twinx()
    param_vals = [params[m] for m in MODELS]
    line = ax2.plot(MODELS, param_vals, 'ko-', linewidth=2, markersize=10)
    ax2.set_ylabel('Parameters (K)', fontsize=12)
    ax2.set_ylim(0, max(param_vals) * 1.3)
    
    for i, (m, p) in enumerate(zip(MODELS, param_vals)):
        ax2.annotate(f'{p}K', (i, p), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    ax1.set_title('Figure 3: Model Complexity vs Performance', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_complexity_performance.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_complexity_performance.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 04_complexity_performance.png")

def create_relative_improvement():
    """相对性能提升"""
    baseline = 'Seq2Point'
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    device_names = ['Dishwasher', 'Kettle', 'Microwave', 'Washingmachine']
    
    for idx, device in enumerate(DEVICES):
        ax = axes[idx]
        baseline_mae = EXPERIMENTS[baseline][device]['MAE']
        improvements = []
        
        for model in MODELS[1:]:
            current_mae = EXPERIMENTS[model][device]['MAE']
            improvement = (baseline_mae - current_mae) / baseline_mae * 100
            improvements.append(improvement)
        
        colors = [COLORS[m] for m in MODELS[1:]]
        bars = ax.bar(MODELS[1:], improvements, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('MAE Reduction (%)', fontsize=10)
        ax.set_title(device_names[idx], fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 4: MAE Reduction Relative to Seq2Point (Higher is Better)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_relative_improvement.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, '05_relative_improvement.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 05_relative_improvement.png")

if __name__ == "__main__":
    print("Generating summary figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 40)
    
    create_summary_table()
    create_mae_comparison()
    create_heatmap()
    create_complexity_chart()
    create_relative_improvement()
    
    print("-" * 40)
    print("Done! All figures saved.")
