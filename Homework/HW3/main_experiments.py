"""
主实验脚本
运行所有实验并生成完整的结果

包含实验：
1. 参数化方式对比（均匀采样 + 非均匀采样）→ 报告4.1节
2. 噪声鲁棒性测试 → 报告4.2节
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from data_generator import *
from parameterization import *
from curve_fitting import ParametricCurveFitter
from metrics import compute_all_metrics, format_metrics
from visualize import *
# fourier_animation 功能已由独立脚本 fourier_star_gui.py 和 fourier_draw_gui.py 实现
from experiment_logger import ExperimentLogger

# 设置随机种子以保证可复现性
np.random.seed(42)

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化实验记录器
logger = ExperimentLogger()


def create_nonuniform_samples(curve_func, n_points=30, **kwargs):
    """
    创建非均匀采样点（在曲率大的地方采样更密）
    
    关键假设：我们假设采样点的顺序已知（按参数t递增排列）
    在实际应用中，如果点顺序未知，需要先用TSP等算法确定顺序
    """
    # 生成密集采样
    dense_points, dense_t = curve_func(n_points=500, **kwargs)
    
    # 非均匀采样：在曲率大的地方采样更密
    # 计算曲率（简化版）
    dx = np.diff(dense_points[:, 0])
    dy = np.diff(dense_points[:, 1])
    curvature_approx = np.abs(np.diff(dx)) + np.abs(np.diff(dy))
    curvature_approx = np.concatenate([[0], curvature_approx, [0]])
    
    # 归一化为采样概率
    prob = curvature_approx + 0.1  # 加小常数避免零概率
    prob = prob / prob.sum()
    
    # 按概率采样
    indices = np.sort(np.random.choice(len(dense_points), n_points, replace=False, p=prob))
    
    return dense_points[indices], dense_points


def experiment_1_parameterization_comparison():
    """
    实验1：对比三种参数化方式
    包含均匀采样和非均匀采样两种情况
    """
    print("\n" + "="*60)
    print("实验1：参数化方式对比")
    print("="*60)
    
    # 测试多种曲线（均匀采样）
    print("\n--- 1a. 均匀采样测试 ---")
    test_curves_uniform = [
        ('Circle', generate_circle, {'n_points': 30}, True),
        ('Heart', generate_heart, {'n_points': 40}, True),
        ('Star', generate_star, {'n_points': 50}, True),
        ('S-Curve', generate_s_curve, {'n_points': 30}, False),
    ]
    
    param_methods = ['uniform', 'chord', 'centripetal']
    
    for curve_name, gen_func, params, is_closed in test_curves_uniform:
        print(f"\n处理曲线: {curve_name}")
        
        # 生成数据
        points, _ = gen_func(**params)
        true_curve, _ = gen_func(n_points=500)
        
        # 对比三种参数化
        curves_dict = {}
        metrics_dict = {}
        
        for method in param_methods:
            # 参数化
            if is_closed:
                t = parameterize_closed_curve(points, method=method)
            else:
                param_func = get_parameterization_function(method)
                t = param_func(points)
            
            # 拟合
            fitter = ParametricCurveFitter(method='cubic_spline', periodic=is_closed)
            fitter.fit(points, t)
            t_eval = np.linspace(0, 1, 500)
            fitted_curve = fitter.evaluate(t_eval)
            
            curves_dict[method.capitalize()] = fitted_curve
            
            # 计算指标
            metrics = compute_all_metrics(points, fitted_curve, true_curve)
            metrics_dict[method.capitalize()] = metrics
            print(f"  {method}: RMSE={metrics['rmse_true']:.6f}")
            
            # 记录到日志
            logger.log_experiment('Exp1_Parameterization', curve_name, 
                                method.capitalize(), metrics,
                                f'均匀采样,n={params["n_points"]}')
        
        # 可视化
        plot_parameterization_comparison(
            points, curves_dict, true_curve,
            title=f'{curve_name}: Parameterization Comparison',
            save_name=f'exp1_{curve_name.lower()}_param_comparison.png'
        )
        
        # 叠加对比
        plot_overlay_comparison(
            points, curves_dict, true_curve,
            title=f'{curve_name}: Overlay Comparison',
            save_name=f'exp1_{curve_name.lower()}_overlay.png'
        )
        
        # 指标对比
        plot_metrics_bar(
            metrics_dict,
            metric_keys=['rmse_true', 'max_dev_samples', 'hausdorff', 'smoothness_energy', 'curvature_std'],
            title=f'{curve_name}: Metrics Comparison',
            save_name=f'exp1_{curve_name.lower()}_metrics.png'
        )
    
    # 非均匀采样测试（更能体现参数化差异）
    print("\n--- 1b. 非均匀采样测试（更能体现参数化差异）---")
    test_curves_nonuniform = [
        ('Heart', generate_heart, {}, True),
        ('Star', generate_star, {}, True),
        ('Random_Blob', generate_random_blob, {}, True),
    ]
    
    for curve_name, gen_func, params, is_closed in test_curves_nonuniform:
        print(f"\n处理曲线: {curve_name} (非均匀采样)")
        
        # 创建非均匀采样
        points, true_curve = create_nonuniform_samples(gen_func, n_points=25, **params)
        
        curves_dict = {}
        metrics_dict = {}
        
        for method in param_methods:
            # 参数化
            if is_closed:
                t = parameterize_closed_curve(points, method=method)
            else:
                param_func = get_parameterization_function(method)
                t = param_func(points)
            
            # 拟合
            fitter = ParametricCurveFitter(method='cubic_spline', periodic=is_closed)
            fitter.fit(points, t)
            t_eval = np.linspace(0, 1, 500)
            fitted_curve = fitter.evaluate(t_eval)
            
            curves_dict[method.capitalize()] = fitted_curve
            
            # 计算指标
            metrics = compute_all_metrics(points, fitted_curve, true_curve)
            metrics_dict[method.capitalize()] = metrics
            print(f"  {method}: RMSE={metrics['rmse_true']:.6f}, "
                  f"Max_Dev={metrics['max_dev_samples']:.6f}")
            
            # 记录到日志
            logger.log_experiment('Exp1_Parameterization_Nonuniform', curve_name, 
                                method.capitalize(), metrics, 
                                f'非均匀采样,n=25')
        
        # 可视化
        plot_parameterization_comparison(
            points, curves_dict, true_curve,
            title=f'{curve_name}: Parameterization Comparison (Non-uniform Sampling)',
            save_name=f'exp1_nonuniform_{curve_name.lower()}_param.png'
        )
        
        plot_overlay_comparison(
            points, curves_dict, true_curve,
            title=f'{curve_name}: Overlay Comparison (Non-uniform)',
            save_name=f'exp1_nonuniform_{curve_name.lower()}_overlay.png'
        )
        
        plot_metrics_bar(
            metrics_dict,
            metric_keys=['rmse_true', 'max_dev_samples', 'hausdorff', 'smoothness_energy', 'curvature_std'],
            title=f'{curve_name}: Metrics Comparison (Non-uniform)',
            save_name=f'exp1_nonuniform_{curve_name.lower()}_metrics.png'
        )


def experiment_2_noise_robustness():
    """
    实验2：噪声鲁棒性测试（插值 vs 平滑拟合）
    """
    print("\n" + "="*60)
    print("实验2：噪声鲁棒性测试")
    print("="*60)
    
    test_curves = [
        ('Heart', generate_heart, {'n_points': 40}, True),
        ('Ellipse', generate_ellipse, {'n_points': 30}, True),
    ]
    
    noise_levels = [0.02, 0.05, 0.10]
    
    for curve_name, gen_func, params, is_closed in test_curves:
        print(f"\n处理曲线: {curve_name}")
        
        # 生成干净数据
        points_clean, _ = gen_func(**params)
        true_curve, _ = gen_func(n_points=500)
        
        for noise_level in noise_levels:
            print(f"  噪声水平: {noise_level}")
            
            # 添加噪声
            points_noisy = add_noise(points_clean, noise_level=noise_level)
            
            # 使用弦长参数化
            t = parameterize_closed_curve(points_noisy, method='chord') if is_closed \
                else chord_length_parameterization(points_noisy)
            
            # 插值
            fitter_interp = ParametricCurveFitter(method='cubic_spline', periodic=is_closed)
            fitter_interp.fit(points_noisy, t)
            t_eval = np.linspace(0, 1, 500)
            curve_interp = fitter_interp.evaluate(t_eval)
            
            # 平滑拟合
            # 根据曲线类型和噪声水平调整平滑参数
            # 简单曲线（如椭圆）需要更小的平滑参数，复杂曲线（如心形）需要更大的平滑参数
            if curve_name == 'Ellipse':
                smoothing = noise_level * 10  # 椭圆使用较小的平滑参数
            else:
                smoothing = noise_level * 100  # 心形等复杂曲线使用较大的平滑参数
            fitter_smooth = ParametricCurveFitter(method='smooth_spline', 
                                                 smoothing=smoothing, periodic=is_closed)
            fitter_smooth.fit(points_noisy, t)
            curve_smooth = fitter_smooth.evaluate(t_eval)
            
            # 计算指标
            metrics_interp = compute_all_metrics(points_noisy, curve_interp, true_curve)
            metrics_smooth = compute_all_metrics(points_noisy, curve_smooth, true_curve)
            
            print(f"    插值 RMSE: {metrics_interp['rmse_true']:.6f}")
            print(f"    平滑 RMSE: {metrics_smooth['rmse_true']:.6f}")
            
            # 记录到日志
            logger.log_experiment('Exp2_Noise_Robustness', curve_name, 
                                'Interpolation', metrics_interp,
                                f'noise={noise_level}')
            logger.log_experiment('Exp2_Noise_Robustness', curve_name, 
                                'Smooth_Fitting', metrics_smooth,
                                f'noise={noise_level},s={smoothing}')
            
            # 可视化
            plot_noise_comparison(
                points_clean, points_noisy, curve_interp, curve_smooth, true_curve,
                title=f'{curve_name} with Noise Level {noise_level}',
                save_name=f'exp2_{curve_name.lower()}_noise_{int(noise_level*100)}.png'
            )


def run_all_experiments():
    """运行所有实验"""
    print("\n" + "="*70)
    print(" "*15 + "平面点列曲线拟合实验")
    print("="*70)
    print("\n包含2组实验：")
    print("  1. 参数化方式对比（均匀+非均匀采样）")
    print("  2. 噪声鲁棒性测试")
    print("="*70)
    
    try:
        experiment_1_parameterization_comparison()
    except Exception as e:
        print(f"实验1出错: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        experiment_2_noise_robustness()
    except Exception as e:
        print(f"实验2出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存实验日志并生成对比表格
    logger.save_session_summary()
    logger.generate_all_comparison_tables()
    logger.print_summary()
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print("="*70)
    print(f"图片结果: figures/ 目录 (30+张图片和动画)")
    print(f"数据表格: experiment_logs/ 目录")
    print(f"  - CSV汇总: experiment_logs/experiment_summary.csv")
    print(f"  - Markdown表格: experiment_logs/*_comparison.md")
    print("="*70)


if __name__ == '__main__':
    run_all_experiments()
