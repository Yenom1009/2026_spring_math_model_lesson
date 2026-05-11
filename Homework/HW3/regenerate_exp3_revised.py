"""
重新生成实验3修订版图片
对比插值方法 vs 拟合方法在不同采样密度下的表现
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from data_generator import generate_heart, generate_random_blob, generate_s_curve
from parameterization import parameterize_closed_curve, chord_length_parameterization
from curve_fitting import ParametricCurveFitter
from metrics import compute_all_metrics
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_sampling_comparison_interp_vs_fit(curve_name, gen_func, is_closed, save_name):
    """
    对比插值方法和拟合方法在不同采样密度下的表现
    上排：插值方法（4种采样密度）
    下排：拟合方法（4种采样密度，s=1.0）
    """
    sampling_densities = [10, 20, 40, 80]
    
    # 生成真实曲线（500点）
    true_curve, _ = gen_func(n_points=500)
    
    # 打印表头
    print(f"\n{curve_name} - 插值方法指标：")
    print(f"{'n':>5} {'RMSE_True':>12} {'Max_Dev':>12} {'Hausdorff':>12} {'Smoothness':>12}")
    print("-" * 60)
    
    interp_metrics_list = []
    fit_metrics_list = []
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for col_idx, n_samples in enumerate(sampling_densities):
        # 从真实曲线中均匀子采样（确保采样点在曲线上）
        indices = np.linspace(0, len(true_curve)-1, n_samples, dtype=int, endpoint=False)
        points = true_curve[indices]
        
        # 参数化
        if is_closed:
            t = parameterize_closed_curve(points, method='chord')
        else:
            t = chord_length_parameterization(points)
        
        t_eval = np.linspace(0, 1, 500)
        
        # 上排：插值方法
        fitter_interp = ParametricCurveFitter(method='cubic_spline', periodic=is_closed)
        fitter_interp.fit(points, t)
        curve_interp = fitter_interp.evaluate(t_eval)
        
        # 计算插值指标
        metrics_interp = compute_all_metrics(points, curve_interp, true_curve)
        interp_metrics_list.append(metrics_interp)
        print(f"{n_samples:>5} {metrics_interp['rmse_true']:>12.6f} {metrics_interp['max_dev_samples']:>12.6f} {metrics_interp['hausdorff']:>12.6f} {metrics_interp['smoothness_energy']:>12.6f}")
        
        ax_interp = axes[0, col_idx]
        ax_interp.plot(true_curve[:, 0], true_curve[:, 1], 'gray', alpha=0.3, linewidth=2, label='True Curve')
        ax_interp.plot(curve_interp[:, 0], curve_interp[:, 1], 'b-', linewidth=2, label='Interpolation')
        ax_interp.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5, label=f'Samples (n={n_samples})')
        ax_interp.set_title(f'Interpolation (n={n_samples})', fontsize=14, fontweight='bold')
        ax_interp.axis('equal')
        ax_interp.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_interp.legend(loc='best', fontsize=10)
        
        # 下排：拟合方法（s=1.0）
        fitter_fit = ParametricCurveFitter(method='smooth_spline', smoothing=1.0, periodic=is_closed)
        fitter_fit.fit(points, t)
        curve_fit = fitter_fit.evaluate(t_eval)
        
        # 计算拟合指标
        metrics_fit = compute_all_metrics(points, curve_fit, true_curve)
        fit_metrics_list.append(metrics_fit)
        
        ax_fit = axes[1, col_idx]
        ax_fit.plot(true_curve[:, 0], true_curve[:, 1], 'gray', alpha=0.3, linewidth=2, label='True Curve')
        ax_fit.plot(curve_fit[:, 0], curve_fit[:, 1], 'g-', linewidth=2, label='Fitting (s=1.0)')
        ax_fit.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5, label=f'Samples (n={n_samples})')
        ax_fit.set_title(f'Fitting s=1.0 (n={n_samples})', fontsize=14, fontweight='bold')
        ax_fit.axis('equal')
        ax_fit.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_fit.legend(loc='best', fontsize=10)
    
    # 打印拟合方法指标
    print(f"\n{curve_name} - 拟合方法指标 (s=1.0)：")
    print(f"{'n':>5} {'RMSE_True':>12} {'Max_Dev':>12} {'Hausdorff':>12} {'Smoothness':>12}")
    print("-" * 60)
    for idx, n_samples in enumerate(sampling_densities):
        m = fit_metrics_list[idx]
        print(f"{n_samples:>5} {m['rmse_true']:>12.6f} {m['max_dev_samples']:>12.6f} {m['hausdorff']:>12.6f} {m['smoothness_energy']:>12.6f}")
    
    # 总标题
    fig.suptitle(f'{curve_name}: Interpolation vs Fitting at Different Sampling Densities', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已生成: {save_name}")


if __name__ == '__main__':
    print("重新生成实验3修订版图片...")
    print("="*70)
    
    # 1. 心形曲线（规则闭合）
    print("\n生成心形曲线图...")
    plot_sampling_comparison_interp_vs_fit(
        'Heart', generate_heart, True, 
        'exp3_revised_sampling_heart.png'
    )
    
    # 2. 复杂不规则闭合曲线（固定12模态随机Blob）
    print("\n生成复杂不规则闭合曲线图（12模态）...")
    # 使用lambda确保每次生成相同的12模态Blob
    def generate_fixed_blob_12modes(n_points=50):
        return generate_random_blob(n_points=n_points, n_modes=12, radius=5.0)
    
    plot_sampling_comparison_interp_vs_fit(
        'Fixed Random Blob (12 modes)', generate_fixed_blob_12modes, True,
        'exp3_revised_sampling_fixed_random_blob.png'
    )
    
    # 3. S曲线（开放曲线）
    print("\n生成S曲线图...")
    plot_sampling_comparison_interp_vs_fit(
        'S-Curve', generate_s_curve, False,
        'exp3_revised_sampling_s-curve.png'
    )
    
    print("\n"+"="*70)
    print("所有图片已重新生成！")
