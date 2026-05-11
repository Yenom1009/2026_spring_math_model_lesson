"""
实验9最终版：多种基函数的插值与拟合对比
////别问为什么是实验九，因为中间实验错了一堆全删了，这是第九ge
对比4种基函数类型：
1. 多项式基函数 (Polynomial) - 分段多项式插值 + 全局最小二乘拟合
2. 三角基函数 (Trigonometric) - FFT插值(均匀参数化) + 截断傅里叶拟合
3. 径向基函数 (RBF) - 严格插值 + 带正则化拟合
4. 样条基函数 (Spline) - 三次样条插值 + 平滑B样条拟合
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, RBFInterpolator, splprep, splev
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from data_generator import generate_heart, generate_star, add_noise
from parameterization import parameterize_closed_curve
from metrics import compute_all_metrics

np.random.seed(42)
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 1. 多项式基函数 ====================

def polynomial_interpolation(points, t, t_eval, periodic=True, seg_degree=3):
    """分段多项式插值 - 每段用低阶多项式，避免Runge现象
    
    数学表达式：在每段[t_i, t_{i+k}]上，
    x(t) = a_0 + a_1*t + a_2*t^2 + ... + a_d*t^d
    y(t) = b_0 + b_1*t + b_2*t^2 + ... + b_d*t^d
    """
    x, y = points[:, 0].copy(), points[:, 1].copy()
    t_fit = t.copy()
    
    if periodic and not np.isclose(x[0], x[-1]):
        t_fit = np.concatenate([t_fit, [1.0]])
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])
    
    n = len(t_fit)
    x_eval = np.zeros(len(t_eval))
    y_eval = np.zeros(len(t_eval))
    
    # 分段：每seg_degree个点一段
    segments = []
    i = 0
    while i < n - 1:
        end = min(i + seg_degree, n - 1)
        segments.append((i, end))
        i = end
    
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        idx = slice(seg_start, seg_end + 1)
        t_seg = t_fit[idx]
        x_seg = x[idx]
        y_seg = y[idx]
        
        t_lo, t_hi = t_seg[0], t_seg[-1]
        if seg_idx == len(segments) - 1:
            mask = (t_eval >= t_lo) & (t_eval <= t_hi)
        else:
            mask = (t_eval >= t_lo) & (t_eval < t_hi)
        
        if np.sum(mask) == 0:
            continue
        
        deg = min(len(t_seg) - 1, seg_degree)
        cx = np.polyfit(t_seg, x_seg, deg)
        cy = np.polyfit(t_seg, y_seg, deg)
        x_eval[mask] = np.polyval(cx, t_eval[mask])
        y_eval[mask] = np.polyval(cy, t_eval[mask])
    
    return np.column_stack([x_eval, y_eval])


def polynomial_fitting(points, t, t_eval, periodic=True, degree=10):
    """全局多项式最小二乘拟合
    
    数学表达式：x(t) = sum_{k=0}^{d} a_k * t^k，最小化 sum_i |x(t_i) - x_i|^2
    """
    x, y = points[:, 0].copy(), points[:, 1].copy()
    t_fit = t.copy()
    
    if periodic and not np.isclose(x[0], x[-1]):
        t_fit = np.concatenate([t_fit, [1.0]])
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])
    
    coeffs_x = np.polyfit(t_fit, x, degree)
    coeffs_y = np.polyfit(t_fit, y, degree)
    
    return np.column_stack([np.polyval(coeffs_x, t_eval), np.polyval(coeffs_y, t_eval)])


# ==================== 2. 三角基函数 ====================

def trigonometric_interpolation(points, t_unused, t_eval, periodic=True):
    """三角插值 - 使用均匀参数化+FFT（标准方法）
    
    对闭合曲线，使用均匀分布的参数值，FFT得到完整傅里叶系数。
    数学表达式：x(t) = sum_{k} c_k * exp(2πi*k*t)
    """
    n = len(points)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    
    # 三角插值的标准做法：使用均匀参数化
    # t_uniform = k/n, k=0,1,...,n-1
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    
    # 在t_eval上重建（t_eval in [0,1]）
    x_eval = np.zeros(len(t_eval))
    y_eval = np.zeros(len(t_eval))
    
    for k in range(n):
        # 频率索引处理
        if k <= n // 2:
            freq = k
        else:
            freq = k - n
        
        phase = 2 * np.pi * freq * t_eval
        x_eval += (fx[k].real * np.cos(phase) - fx[k].imag * np.sin(phase)) / n
        y_eval += (fy[k].real * np.cos(phase) - fy[k].imag * np.sin(phase)) / n
    
    return np.column_stack([x_eval, y_eval])


def trigonometric_fitting(points, t, t_eval, periodic=True, n_terms=5):
    """截断傅里叶级数拟合 - 只保留低频分量
    
    数学表达式：x(t) ≈ a_0/2 + sum_{k=1}^{K} [a_k cos(2πkt) + b_k sin(2πkt)]
    """
    x, y = points[:, 0].copy(), points[:, 1].copy()
    n = len(x)
    
    n_basis = 2 * n_terms + 1
    A = np.zeros((n, n_basis))
    A[:, 0] = 1.0
    
    col = 1
    for k in range(1, n_terms + 1):
        A[:, col] = np.cos(2 * np.pi * k * t)
        col += 1
        A[:, col] = np.sin(2 * np.pi * k * t)
        col += 1
    
    coeffs_x = np.linalg.lstsq(A, x, rcond=None)[0]
    coeffs_y = np.linalg.lstsq(A, y, rcond=None)[0]
    
    A_eval = np.zeros((len(t_eval), n_basis))
    A_eval[:, 0] = 1.0
    col = 1
    for k in range(1, n_terms + 1):
        A_eval[:, col] = np.cos(2 * np.pi * k * t_eval)
        col += 1
        A_eval[:, col] = np.sin(2 * np.pi * k * t_eval)
        col += 1
    
    return np.column_stack([A_eval @ coeffs_x, A_eval @ coeffs_y])


# ==================== 3. 径向基函数 ====================

def rbf_interpolation(points, t, t_eval, periodic=True, kernel='thin_plate_spline'):
    """RBF严格插值 - smoothing=0
    
    数学表达式：x(t) = sum_{i=1}^{n} w_i * phi(|t - t_i|)
    其中phi是径向基函数（薄板样条：phi(r) = r^2 * log(r)）
    """
    t_col = t.reshape(-1, 1)
    t_eval_col = t_eval.reshape(-1, 1)
    
    rbf_x = RBFInterpolator(t_col, points[:, 0], kernel=kernel, smoothing=0.0)
    rbf_y = RBFInterpolator(t_col, points[:, 1], kernel=kernel, smoothing=0.0)
    
    return np.column_stack([rbf_x(t_eval_col), rbf_y(t_eval_col)])


def rbf_fitting(points, t, t_eval, periodic=True, kernel='thin_plate_spline', smoothing=0.01):
    """RBF正则化拟合
    
    数学表达式同上，但最小化 sum|f(t_i)-y_i|^2 + λ*||f||^2
    """
    t_col = t.reshape(-1, 1)
    t_eval_col = t_eval.reshape(-1, 1)
    
    rbf_x = RBFInterpolator(t_col, points[:, 0], kernel=kernel, smoothing=smoothing)
    rbf_y = RBFInterpolator(t_col, points[:, 1], kernel=kernel, smoothing=smoothing)
    
    return np.column_stack([rbf_x(t_eval_col), rbf_y(t_eval_col)])


# ==================== 4. 样条基函数 ====================

def spline_interpolation(points, t, t_eval, periodic=True):
    """三次样条插值 - 严格通过所有数据点
    
    数学表达式：在每段[t_i, t_{i+1}]上，
    S_i(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3
    满足C2连续性条件
    """
    x, y = points[:, 0].copy(), points[:, 1].copy()
    t_fit = t.copy()
    
    if periodic:
        if not np.isclose(t_fit[-1], 1.0) or not np.isclose(x[0], x[-1]):
            t_fit = np.concatenate([t_fit, [1.0]])
            x = np.concatenate([x, [x[0]]])
            y = np.concatenate([y, [y[0]]])
        cs_x = CubicSpline(t_fit, x, bc_type='periodic')
        cs_y = CubicSpline(t_fit, y, bc_type='periodic')
    else:
        cs_x = CubicSpline(t_fit, x)
        cs_y = CubicSpline(t_fit, y)
    
    return np.column_stack([cs_x(t_eval), cs_y(t_eval)])


def spline_fitting(points, t, t_eval, periodic=True, smoothing=1.0):
    """平滑B样条拟合 - 允许偏离数据点
    
    最小化 sum|S(t_i)-P_i|^2 + s * integral(S''(t)^2 dt)
    """
    x, y = points[:, 0].copy(), points[:, 1].copy()
    t_fit = t.copy()
    
    if periodic:
        if not np.isclose(t_fit[-1], 1.0) or not np.isclose(x[0], x[-1]):
            t_fit = np.concatenate([t_fit, [1.0]])
            x = np.concatenate([x, [x[0]]])
            y = np.concatenate([y, [y[0]]])
    
    per = 1 if periodic else 0
    tck, _ = splprep([x, y], u=t_fit, s=smoothing, k=3, per=per)
    curve = splev(t_eval, tck)
    return np.column_stack(curve)


# ==================== 主实验 ====================

def run_experiment_9():
    print("\n" + "="*80)
    print("实验9：多种基函数的插值与拟合对比")
    print("="*80)
    
    test_cases = [
        ('heart_clean', generate_heart, {'n_points': 30}, 0.0, '心形曲线（无噪声）'),
        ('heart_noise2', generate_heart, {'n_points': 30}, 0.02, '心形曲线（2%噪声）'),
        ('heart_noise5', generate_heart, {'n_points': 30}, 0.05, '心形曲线（5%噪声）'),
        ('star_clean', generate_star, {'n_points': 40}, 0.0, '星形曲线（无噪声）'),
    ]
    
    methods = {
        'Poly_Interp': {
            'func': lambda pts, t, te: polynomial_interpolation(pts, t, te, periodic=True, seg_degree=3),
            'type': 'interpolation', 'name': '多项式插值(分段3阶)', 'basis': 'Polynomial'
        },
        'Trig_Interp': {
            'func': lambda pts, t, te: trigonometric_interpolation(pts, t, te, periodic=True),
            'type': 'interpolation', 'name': '三角插值(FFT)', 'basis': 'Trigonometric'
        },
        'RBF_Interp': {
            'func': lambda pts, t, te: rbf_interpolation(pts, t, te, periodic=True),
            'type': 'interpolation', 'name': 'RBF插值', 'basis': 'RBF'
        },
        'Spline_Interp': {
            'func': lambda pts, t, te: spline_interpolation(pts, t, te, periodic=True),
            'type': 'interpolation', 'name': '样条插值', 'basis': 'Spline'
        },
        'Poly_Fit': {
            'func': lambda pts, t, te: polynomial_fitting(pts, t, te, periodic=True, degree=10),
            'type': 'fitting', 'name': '多项式拟合(10阶)', 'basis': 'Polynomial'
        },
        'Trig_Fit': {
            'func': lambda pts, t, te: trigonometric_fitting(pts, t, te, periodic=True, n_terms=5),
            'type': 'fitting', 'name': '三角拟合(5项)', 'basis': 'Trigonometric'
        },
        'RBF_Fit': {
            'func': lambda pts, t, te: rbf_fitting(pts, t, te, periodic=True, smoothing=0.01),
            'type': 'fitting', 'name': 'RBF拟合(λ=0.01)', 'basis': 'RBF'
        },
        'Spline_Fit': {
            'func': lambda pts, t, te: spline_fitting(pts, t, te, periodic=True, smoothing=1.0),
            'type': 'fitting', 'name': '样条拟合(s=1)', 'basis': 'Spline'
        },
    }
    
    all_results = {}
    
    for case_name, gen_func, params, noise_level, description in test_cases:
        print(f"\n{'='*60}")
        print(f"场景: {description}")
        print(f"{'='*60}")
        
        points_clean, _ = gen_func(**params)
        true_curve, _ = gen_func(n_points=500)
        
        if noise_level > 0:
            points = add_noise(points_clean, noise_level=noise_level)
        else:
            points = points_clean
        
        t = parameterize_closed_curve(points, method='chord')
        t_eval = np.linspace(0, 1, 500)
        
        results = {}
        for method_key, method_info in methods.items():
            try:
                curve = method_info['func'](points, t, t_eval)
                
                if np.any(np.isnan(curve)) or np.any(np.isinf(curve)):
                    print(f"  {method_info['name']}: NaN/Inf，跳过")
                    continue
                
                max_val = np.max(np.abs(curve))
                if max_val > 50:
                    print(f"  {method_info['name']}: 极端值(max={max_val:.1f})，跳过")
                    continue
                
                metrics = compute_all_metrics(points, curve, true_curve)
                results[method_key] = {
                    'curve': curve, 'metrics': metrics, 'info': method_info
                }
                
                print(f"  {method_info['name']:20s}: RMSE={metrics['rmse_true']:.4f}, "
                      f"Hausdorff={metrics['hausdorff']:.4f}, "
                      f"平滑={metrics['smoothness_energy']:.6f}")
                
            except Exception as e:
                print(f"  {method_info['name']}: 失败 - {str(e)[:60]}")
        
        all_results[case_name] = {
            'results': results, 'points': points, 'true_curve': true_curve,
            'description': description
        }
        
        if len(results) == 0:
            continue
        
        # ===== 可视化1：2x4网格 =====
        basis_types = ['Polynomial', 'Trigonometric', 'RBF', 'Spline']
        basis_labels = ['多项式', '三角函数', 'RBF', '样条']
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for col_idx, (basis, blabel) in enumerate(zip(basis_types, basis_labels)):
            for row_idx, mtype in enumerate(['interpolation', 'fitting']):
                ax = axes[row_idx, col_idx]
                keys = [k for k, v in results.items() 
                       if v['info']['basis'] == basis and v['info']['type'] == mtype]
                
                if keys:
                    key = keys[0]
                    curve = results[key]['curve']
                    m = results[key]['metrics']
                    
                    ax.plot(true_curve[:, 0], true_curve[:, 1], 'g-', 
                           alpha=0.3, linewidth=3, label='真实曲线')
                    color = 'royalblue' if mtype == 'interpolation' else 'darkorange'
                    ax.plot(curve[:, 0], curve[:, 1], '-', color=color,
                           linewidth=2, label='重建曲线')
                    ax.scatter(points[:, 0], points[:, 1], c='red', s=25, 
                             zorder=5, label='采样点')
                    
                    ax.set_title(f'{results[key]["info"]["name"]}\n'
                               f'RMSE={m["rmse_true"]:.4f}', fontsize=10)
                    ax.legend(fontsize=7, loc='best')
                    ax.axis('equal')
                    ax.grid(True, alpha=0.3)
                    
                    pad = 0.5
                    ax.set_xlim([true_curve[:, 0].min()-pad, true_curve[:, 0].max()+pad])
                    ax.set_ylim([true_curve[:, 1].min()-pad, true_curve[:, 1].max()+pad])
                else:
                    ax.text(0.5, 0.5, f'{blabel}\n{"插值" if mtype=="interpolation" else "拟合"}失败',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_facecolor('#f0f0f0')
        
        axes[0, 0].set_ylabel('插值方法\n(通过数据点)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('拟合方法\n(允许偏离)', fontsize=11, fontweight='bold')
        
        for col_idx, blabel in enumerate(basis_labels):
            axes[0, col_idx].text(0.5, 1.15, blabel + '基函数',
                                 transform=axes[0, col_idx].transAxes,
                                 ha='center', fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.suptitle(f'不同基函数的插值与拟合对比 — {description}', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'exp9_final_comparison_{case_name}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # ===== 可视化2：指标柱状图 =====
        fig, axes_m = plt.subplots(1, 3, figsize=(18, 6))
        
        names = [results[k]['info']['name'] for k in results]
        types = [results[k]['info']['type'] for k in results]
        colors = ['steelblue' if t == 'interpolation' else 'darkorange' for t in types]
        
        for ax_idx, (metric_key, metric_name) in enumerate([
            ('rmse_true', 'RMSE (vs 真实曲线)'),
            ('hausdorff', 'Hausdorff距离'),
            ('smoothness_energy', '平滑能量')
        ]):
            values = [results[k]['metrics'][metric_key] for k in results]
            bars = axes_m[ax_idx].barh(range(len(names)), values, color=colors)
            axes_m[ax_idx].set_yticks(range(len(names)))
            axes_m[ax_idx].set_yticklabels(names, fontsize=8)
            axes_m[ax_idx].set_title(metric_name, fontsize=12)
            axes_m[ax_idx].invert_yaxis()
            axes_m[ax_idx].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'性能指标对比 — {description}\n■蓝色=插值  ■橙色=拟合',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'exp9_final_metrics_{case_name}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # ===== 可视化3：节点数影响实验 =====
    print(f"\n{'='*60}")
    print("节点数对插值效果的影响")
    print(f"{'='*60}")
    
    n_points_list = [10, 15, 20, 30, 50, 80]
    fig, axes = plt.subplots(2, len(n_points_list), figsize=(24, 10))
    
    true_curve, _ = generate_heart(n_points=500)
    
    for col_idx, n_pts in enumerate(n_points_list):
        pts_clean, _ = generate_heart(n_points=n_pts)
        t_pts = parameterize_closed_curve(pts_clean, method='chord')
        t_ev = np.linspace(0, 1, 500)
        
        # 样条插值
        try:
            curve_spline = spline_interpolation(pts_clean, t_pts, t_ev)
            m_spline = compute_all_metrics(pts_clean, curve_spline, true_curve)
            
            axes[0, col_idx].plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.3, linewidth=3)
            axes[0, col_idx].plot(curve_spline[:, 0], curve_spline[:, 1], 'b-', linewidth=2)
            axes[0, col_idx].scatter(pts_clean[:, 0], pts_clean[:, 1], c='red', s=20, zorder=5)
            axes[0, col_idx].set_title(f'样条插值 n={n_pts}\nRMSE={m_spline["rmse_true"]:.4f}')
            axes[0, col_idx].axis('equal')
            axes[0, col_idx].grid(True, alpha=0.3)
        except:
            axes[0, col_idx].text(0.5, 0.5, '失败', ha='center', va='center',
                                 transform=axes[0, col_idx].transAxes)
        
        # 三角插值(FFT)
        try:
            curve_trig = trigonometric_interpolation(pts_clean, t_pts, t_ev)
            m_trig = compute_all_metrics(pts_clean, curve_trig, true_curve)
            
            axes[1, col_idx].plot(true_curve[:, 0], true_curve[:, 1], 'g-', alpha=0.3, linewidth=3)
            axes[1, col_idx].plot(curve_trig[:, 0], curve_trig[:, 1], 'darkorange', linewidth=2)
            axes[1, col_idx].scatter(pts_clean[:, 0], pts_clean[:, 1], c='red', s=20, zorder=5)
            axes[1, col_idx].set_title(f'三角插值 n={n_pts}\nRMSE={m_trig["rmse_true"]:.4f}')
            axes[1, col_idx].axis('equal')
            axes[1, col_idx].grid(True, alpha=0.3)
        except:
            axes[1, col_idx].text(0.5, 0.5, '失败', ha='center', va='center',
                                 transform=axes[1, col_idx].transAxes)
    
    axes[0, 0].set_ylabel('样条插值', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('三角插值', fontsize=12, fontweight='bold')
    plt.suptitle('节点数对不同基函数插值效果的影响 — 心形曲线',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp9_final_node_count.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n实验9完成！")


if __name__ == '__main__':
    run_experiment_9()
