"""
可视化模块
生成各种实验对比图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_single_curve(points, fitted_curve, true_curve=None, title='', 
                      save_name=None, show=False):
    """
    绘制单组拟合结果
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if true_curve is not None:
        ax.plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.4, 
                linewidth=3, label='True curve')
    
    ax.plot(fitted_curve[:, 0], fitted_curve[:, 1], 'b-', linewidth=2, 
            label='Fitted curve')
    ax.scatter(points[:, 0], points[:, 1], c='red', s=40, zorder=5, 
              edgecolors='black', linewidths=0.5, label='Sample points')
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_parameterization_comparison(points, curves_dict, true_curve=None, 
                                     title='', save_name=None, show=False):
    """
    对比三种参数化方式的拟合结果
    
    参数:
        points: 采样点 (n, 2)
        curves_dict: {'method_name': fitted_curve, ...}
        true_curve: 真实曲线 (m, 2)
    """
    n_methods = len(curves_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 7))
    if n_methods == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, curve) in enumerate(curves_dict.items()):
        ax = axes[i]
        
        if true_curve is not None:
            ax.plot(true_curve[:, 0], true_curve[:, 1], 'gray', alpha=0.3,
                    linewidth=3, label='True curve')
        
        ax.plot(curve[:, 0], curve[:, 1], color=colors[i % len(colors)], 
                linewidth=2, label=f'{name}')
        ax.scatter(points[:, 0], points[:, 1], c='red', s=30, zorder=5,
                  edgecolors='black', linewidths=0.5, label='Samples')
        
        ax.set_title(name, fontsize=13)
        ax.legend(fontsize=10)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_overlay_comparison(points, curves_dict, true_curve=None,
                            title='', save_name=None, show=False):
    """
    在同一张图中叠加不同方法的拟合结果
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if true_curve is not None:
        ax.plot(true_curve[:, 0], true_curve[:, 1], 'k-', alpha=0.2,
                linewidth=4, label='True curve')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (name, curve) in enumerate(curves_dict.items()):
        ax.plot(curve[:, 0], curve[:, 1], color=colors[i % len(colors)],
                linewidth=2, label=name)
    
    ax.scatter(points[:, 0], points[:, 1], c='red', s=40, zorder=5,
              edgecolors='black', linewidths=0.5, label='Samples')
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_noise_comparison(points_clean, points_noisy, curve_interp, curve_smooth,
                          true_curve=None, title='', save_name=None, show=False):
    """
    对比插值与平滑拟合在噪声下的表现
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：插值
    ax = axes[0]
    if true_curve is not None:
        ax.plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.4,
                linewidth=3, label='True curve')
    ax.plot(curve_interp[:, 0], curve_interp[:, 1], 'b-', linewidth=2,
            label='Interpolation')
    ax.scatter(points_noisy[:, 0], points_noisy[:, 1], c='red', s=30,
              zorder=5, edgecolors='black', linewidths=0.5, label='Noisy samples')
    ax.set_title('Cubic Spline Interpolation (with noise)', fontsize=13)
    ax.legend(fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # 右图：平滑拟合
    ax = axes[1]
    if true_curve is not None:
        ax.plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.4,
                linewidth=3, label='True curve')
    ax.plot(curve_smooth[:, 0], curve_smooth[:, 1], 'b-', linewidth=2,
            label='Smooth fitting')
    ax.scatter(points_noisy[:, 0], points_noisy[:, 1], c='red', s=30,
              zorder=5, edgecolors='black', linewidths=0.5, label='Noisy samples')
    ax.set_title('Smooth Spline Fitting (with noise)', fontsize=13)
    ax.legend(fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_sampling_density_comparison(curves_by_density, true_curve=None,
                                     title='', save_name=None, show=False):
    """
    对比不同采样密度下的拟合效果
    
    参数:
        curves_by_density: {n_samples: (points, fitted_curve), ...}
    """
    n = len(curves_by_density)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 7))
    if n == 1:
        axes = [axes]
    
    for i, (n_samples, (points, curve)) in enumerate(curves_by_density.items()):
        ax = axes[i]
        if true_curve is not None:
            ax.plot(true_curve[:, 0], true_curve[:, 1], 'gray', alpha=0.3,
                    linewidth=3, label='True curve')
        ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, label='Fitted')
        ax.scatter(points[:, 0], points[:, 1], c='red', s=30, zorder=5,
                  edgecolors='black', linewidths=0.5, label='Samples')
        ax.set_title(f'n = {n_samples}', fontsize=13)
        ax.legend(fontsize=10)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics_bar(metrics_dict, metric_keys=None, title='', 
                     save_name=None, show=False):
    """
    用柱状图对比不同方法的指标
    
    参数:
        metrics_dict: {'method_name': metrics_dict, ...}
        metric_keys: 要展示的指标名列表
    """
    if metric_keys is None:
        # 使用第一个方法的指标
        first_metrics = next(iter(metrics_dict.values()))
        metric_keys = list(first_metrics.keys())
    
    n_metrics = len(metric_keys)
    n_methods = len(metrics_dict)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    method_names = list(metrics_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for i, key in enumerate(metric_keys):
        ax = axes[i]
        values = [metrics_dict[name].get(key, 0) for name in method_names]
        bars = ax.bar(range(n_methods), values, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax.set_title(key.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_all_test_curves(curves_data, save_name='all_test_curves.png', show=False):
    """
    绘制所有测试曲线概览
    
    参数:
        curves_data: [(name, points, true_curve), ...]
    """
    n = len(curves_data)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (name, points, true_curve) in enumerate(curves_data):
        ax = axes[i]
        if true_curve is not None:
            ax.plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.5,
                    linewidth=2, label='True')
        ax.scatter(points[:, 0], points[:, 1], c='red', s=15, label='Samples')
        ax.set_title(name, fontsize=12)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if save_name:
        fig.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
