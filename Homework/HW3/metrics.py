"""
误差评估模块
计算拟合曲线与真实曲线/采样点之间的误差
"""
import numpy as np
from scipy.spatial.distance import cdist


def point_to_curve_distance(points, curve):
    """
    计算采样点到拟合曲线的最近距离
    
    参数:
        points: 采样点 (n, 2)
        curve: 拟合曲线离散点 (m, 2)
    返回:
        distances: 每个采样点到曲线的最近距离 (n,)
    """
    dist_matrix = cdist(points, curve)
    distances = np.min(dist_matrix, axis=1)
    return distances


def hausdorff_distance(curve1, curve2):
    """
    计算两条曲线之间的 Hausdorff 距离
    
    参数:
        curve1: 曲线1离散点 (n, 2)
        curve2: 曲线2离散点 (m, 2)
    返回:
        h_dist: Hausdorff 距离
    """
    dist_matrix = cdist(curve1, curve2)
    d12 = np.max(np.min(dist_matrix, axis=1))
    d21 = np.max(np.min(dist_matrix, axis=0))
    return max(d12, d21)


def mean_distance(curve1, curve2):
    """
    计算两条曲线之间的平均最近距离（双向）
    
    参数:
        curve1: 曲线1离散点 (n, 2)
        curve2: 曲线2离散点 (m, 2)
    返回:
        avg_dist: 平均最近距离
    """
    dist_matrix = cdist(curve1, curve2)
    d12 = np.mean(np.min(dist_matrix, axis=1))
    d21 = np.mean(np.min(dist_matrix, axis=0))
    return (d12 + d21) / 2


def rmse_distance(points, curve):
    """
    计算采样点到拟合曲线的均方根误差
    
    参数:
        points: 采样点 (n, 2)
        curve: 拟合曲线离散点 (m, 2)
    返回:
        rmse: 均方根误差
    """
    distances = point_to_curve_distance(points, curve)
    return np.sqrt(np.mean(distances**2))


def max_deviation(points, curve):
    """
    计算采样点到拟合曲线的最大偏差
    
    参数:
        points: 采样点 (n, 2)
        curve: 拟合曲线离散点 (m, 2)
    返回:
        max_dev: 最大偏差
    """
    distances = point_to_curve_distance(points, curve)
    return np.max(distances)


def discrete_curvature(curve):
    """
    计算离散曲率（用于评估曲线平滑性）
    
    参数:
        curve: 曲线离散点 (n, 2)
    返回:
        curvature: 离散曲率 (n-2,)
    """
    # 使用三点公式计算曲率
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    
    # 取中间点的一阶导数
    dx_mid = (dx[:-1] + dx[1:]) / 2
    dy_mid = (dy[:-1] + dy[1:]) / 2
    
    # 曲率公式: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx_mid * ddy - dy_mid * ddx)
    denominator = (dx_mid**2 + dy_mid**2) ** 1.5
    
    # 避免除零
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)
    return curvature


def smoothness_energy(curve):
    """
    计算曲线的平滑能量（二阶差分能量）
    
    参数:
        curve: 曲线离散点 (n, 2)
    返回:
        energy: 平滑能量值
    """
    # 二阶差分
    d2x = np.diff(curve[:, 0], n=2)
    d2y = np.diff(curve[:, 1], n=2)
    energy = np.sum(d2x**2 + d2y**2)
    return energy


def compute_all_metrics(sample_points, fitted_curve, true_curve=None):
    """
    计算所有评估指标
    
    参数:
        sample_points: 采样点 (n, 2)
        fitted_curve: 拟合曲线离散点 (m, 2)
        true_curve: 真实曲线离散点 (k, 2)，可选
    返回:
        metrics: 指标字典
    """
    metrics = {}
    
    # 采样点到拟合曲线的误差
    metrics['rmse_samples'] = rmse_distance(sample_points, fitted_curve)
    metrics['max_dev_samples'] = max_deviation(sample_points, fitted_curve)
    metrics['mean_dist_samples'] = np.mean(point_to_curve_distance(sample_points, fitted_curve))
    
    # 平滑性
    metrics['smoothness_energy'] = smoothness_energy(fitted_curve)
    curvature = discrete_curvature(fitted_curve)
    metrics['curvature_std'] = np.std(curvature)
    metrics['curvature_max'] = np.max(curvature)
    
    # 如果有真实曲线
    if true_curve is not None:
        metrics['hausdorff'] = hausdorff_distance(fitted_curve, true_curve)
        metrics['mean_dist_true'] = mean_distance(fitted_curve, true_curve)
        metrics['rmse_true'] = rmse_distance(true_curve, fitted_curve)
    
    return metrics


def format_metrics(metrics, name=""):
    """格式化输出指标"""
    lines = []
    if name:
        lines.append(f"=== {name} ===")
    for key, value in metrics.items():
        lines.append(f"  {key}: {value:.6f}")
    return "\n".join(lines)
