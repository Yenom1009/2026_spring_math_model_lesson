"""
参数化模块
实现三种参数化方式：均匀参数化、弦长参数化、中心参数化
"""
import numpy as np


def uniform_parameterization(points):
    """
    均匀参数化
    参数 t_i 均匀分布在 [0, 1]
    
    参数:
        points: 采样点 (n, 2)
    返回:
        t: 参数值 (n,)
    """
    n = len(points)
    t = np.linspace(0, 1, n)
    return t


def chord_length_parameterization(points):
    """
    弦长参数化
    参数 t_i 与累计弦长成比例
    
    参数:
        points: 采样点 (n, 2)
    返回:
        t: 参数值 (n,)
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # 计算相邻点之间的距离
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    
    # 累计距离
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # 归一化到 [0, 1]
    total_length = cumulative_distances[-1]
    if total_length == 0:
        return uniform_parameterization(points)
    
    t = cumulative_distances / total_length
    return t


def centripetal_parameterization(points, alpha=0.5):
    """
    中心参数化（向心参数化）
    参数 t_i 与累计距离的 alpha 次方成比例
    alpha=0.5 是常用选择，介于均匀和弦长之间
    
    参数:
        points: 采样点 (n, 2)
        alpha: 幂次参数，通常取 0.5
    返回:
        t: 参数值 (n,)
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # 计算相邻点之间的距离
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    
    # 对距离取 alpha 次方
    distances_alpha = distances ** alpha
    
    # 累计
    cumulative = np.concatenate([[0], np.cumsum(distances_alpha)])
    
    # 归一化到 [0, 1]
    total = cumulative[-1]
    if total == 0:
        return uniform_parameterization(points)
    
    t = cumulative / total
    return t


def parameterize_closed_curve(points, method='chord'):
    """
    对闭合曲线进行参数化
    确保首尾参数值满足周期性要求
    
    参数:
        points: 采样点 (n, 2)，假设首尾不重复
        method: 参数化方法 'uniform', 'chord', 'centripetal'
    返回:
        t: 参数值 (n,)
    """
    # 临时添加首点到末尾以计算完整周长
    points_closed = np.vstack([points, points[0:1]])
    
    if method == 'uniform':
        t = uniform_parameterization(points_closed)[:-1]
    elif method == 'chord':
        t = chord_length_parameterization(points_closed)[:-1]
    elif method == 'centripetal':
        t = centripetal_parameterization(points_closed)[:-1]
    else:
        raise ValueError(f"Unknown parameterization method: {method}")
    
    return t


def get_parameterization_function(method):
    """根据方法名返回参数化函数"""
    methods = {
        'uniform': uniform_parameterization,
        'chord': chord_length_parameterization,
        'centripetal': centripetal_parameterization
    }
    return methods.get(method)


if __name__ == '__main__':
    # 测试参数化方法
    import matplotlib.pyplot as plt
    from data_generator import generate_heart
    
    # 生成测试数据：非均匀采样的心形线
    points, _ = generate_heart(n_points=20)
    
    # 人为制造非均匀采样
    indices = np.sort(np.random.choice(len(points), 15, replace=False))
    points_nonuniform = points[indices]
    
    # 三种参数化
    t_uniform = uniform_parameterization(points_nonuniform)
    t_chord = chord_length_parameterization(points_nonuniform)
    t_centripetal = centripetal_parameterization(points_nonuniform)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [
        ('Uniform', t_uniform),
        ('Chord Length', t_chord),
        ('Centripetal', t_centripetal)
    ]
    
    for ax, (name, t) in zip(axes, methods):
        # 绘制点
        scatter = ax.scatter(points_nonuniform[:, 0], points_nonuniform[:, 1], 
                           c=t, cmap='viridis', s=100, edgecolors='black', linewidths=1.5)
        ax.plot(points_nonuniform[:, 0], points_nonuniform[:, 1], 
               'gray', alpha=0.3, linewidth=1)
        
        # 标注参数值
        for i, (x, y, ti) in enumerate(zip(points_nonuniform[:, 0], 
                                           points_nonuniform[:, 1], t)):
            ax.annotate(f'{ti:.2f}', (x, y), fontsize=8, 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title(f'{name} Parameterization')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Parameter t')
    
    plt.tight_layout()
    plt.savefig('test_parameterization.png', dpi=150, bbox_inches='tight')
    print("参数化测试结果已保存到 test_parameterization.png")
