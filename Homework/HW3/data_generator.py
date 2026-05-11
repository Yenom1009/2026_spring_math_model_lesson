"""
数据生成模块
生成各种测试曲线的采样点，支持添加噪声
"""
import numpy as np


def generate_circle(n_points=50, radius=1.0, center=(0, 0)):
    """生成圆形采样点"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack([x, y]), t


def generate_ellipse(n_points=50, a=2.0, b=1.0, center=(0, 0)):
    """生成椭圆采样点"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = center[0] + a * np.cos(t)
    y = center[1] + b * np.sin(t)
    return np.column_stack([x, y]), t


def generate_heart(n_points=100):
    """生成心形线采样点"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return np.column_stack([x, y]), t


def generate_star(n_points=100, n_peaks=5, outer_radius=2.0, inner_radius=1.0):
    """生成星形线采样点"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # 使用极坐标生成星形
    r = outer_radius + (inner_radius - outer_radius) * (1 + np.cos(n_peaks * t)) / 2
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y]), t


def generate_lemniscate(n_points=100, a=1.0):
    """生成双纽线（∞形）采样点"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    # 使用参数方程
    x = a * np.cos(t) / (1 + np.sin(t)**2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    return np.column_stack([x, y]), t


def generate_s_curve(n_points=50):
    """生成S形开放曲线采样点"""
    t = np.linspace(0, 1, n_points)
    x = t
    y = np.sin(2 * np.pi * t)
    return np.column_stack([x, y]), t


def generate_spiral(n_points=100, n_turns=2):
    """生成螺旋线采样点"""
    t = np.linspace(0, 1, n_points)
    theta = 2 * np.pi * n_turns * t
    r = t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y]), t


def generate_wave(n_points=50, n_waves=3):
    """生成波浪线采样点"""
    t = np.linspace(0, 1, n_points)
    x = t
    y = 0.3 * np.sin(2 * np.pi * n_waves * t)
    return np.column_stack([x, y]), t


def add_noise(points, noise_level=0.05):
    """
    为采样点添加高斯噪声
    
    参数:
        points: 原始采样点 (n, 2)
        noise_level: 噪声水平（相对于数据范围的比例）
    """
    data_range = np.ptp(points, axis=0)  # 数据范围
    noise = np.random.randn(*points.shape) * noise_level * data_range
    return points + noise


def subsample_points(points, n_samples):
    """
    对点列进行子采样
    
    参数:
        points: 原始点列 (n, 2)
        n_samples: 目标采样点数
    """
    n_original = len(points)
    if n_samples >= n_original:
        return points
    indices = np.linspace(0, n_original-1, n_samples, dtype=int)
    return points[indices]


def generate_random_polygon(n_points=50, n_vertices=8, radius=5.0):
    """生成随机多边形（不规则图形）"""
    angles = np.sort(np.random.rand(n_vertices) * 2 * np.pi)
    radii = radius * (0.5 + np.random.rand(n_vertices))
    vertices_x = radii * np.cos(angles)
    vertices_y = radii * np.sin(angles)
    vertices_x = np.concatenate([vertices_x, [vertices_x[0]]])
    vertices_y = np.concatenate([vertices_y, [vertices_y[0]]])
    t_vertices = np.linspace(0, 1, n_vertices + 1)
    t = np.linspace(0, 1, n_points)
    x = np.interp(t, t_vertices, vertices_x)
    y = np.interp(t, t_vertices, vertices_y)
    return np.column_stack([x, y]), t


def generate_random_blob(n_points=50, n_modes=6, radius=5.0):
    """生成随机不规则闭合曲线"""
    t = np.linspace(0, 1, n_points, endpoint=False)  # 不包含终点
    theta = 2 * np.pi * t
    r = radius * np.ones_like(theta)
    for k in range(1, n_modes + 1):
        amp = radius * 0.3 * np.random.rand() / k
        phase = 2 * np.pi * np.random.rand()
        r += amp * np.cos(k * theta + phase)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y]), t


def generate_perturbed_circle(n_points=50, n_bumps=5, bump_size=0.3):
    """生成带凸起/凹陷的不规则圆形"""
    t = np.linspace(0, 1, n_points, endpoint=False)  # 不包含终点
    theta = 2 * np.pi * t
    radius = 5.0
    r = radius * np.ones_like(theta)
    for _ in range(n_bumps):
        center = 2 * np.pi * np.random.rand()
        width = 0.5 + np.random.rand()
        height = bump_size * radius * (0.5 + np.random.rand())
        sign = 1 if np.random.rand() > 0.5 else -1
        r += sign * height * np.exp(-((theta - center) / width) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y]), t


def get_curve_function(curve_name):
    """根据曲线名称返回生成函数"""
    curves = {
        'circle': generate_circle,
        'ellipse': generate_ellipse,
        'heart': generate_heart,
        'star': generate_star,
        'lemniscate': generate_lemniscate,
        's_curve': generate_s_curve,
        'spiral': generate_spiral,
        'wave': generate_wave,
        'random_polygon': generate_random_polygon,
        'random_blob': generate_random_blob,
        'perturbed_circle': generate_perturbed_circle
    }
    return curves.get(curve_name)


if __name__ == '__main__':
    # 测试数据生成
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    curves = ['circle', 'ellipse', 'heart', 'star', 
              'lemniscate', 's_curve', 'spiral', 'wave']
    
    for i, curve_name in enumerate(curves):
        func = get_curve_function(curve_name)
        points, _ = func()
        
        axes[i].plot(points[:, 0], points[:, 1], 'b-', alpha=0.5, label='Original')
        axes[i].scatter(points[:, 0], points[:, 1], c='red', s=20, label='Samples')
        axes[i].set_title(curve_name.capitalize())
        axes[i].axis('equal')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_curves.png', dpi=150, bbox_inches='tight')
    print("测试曲线已保存到 test_curves.png")
