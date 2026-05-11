"""
曲线拟合模块
实现三次样条插值和平滑样条拟合
"""
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline, splprep, splev


class ParametricCurveFitter:
    """参数曲线拟合器"""
    
    def __init__(self, method='cubic_spline', smoothing=None, periodic=False):
        self.method = method
        self.smoothing = smoothing
        self.periodic = periodic
        self.spline_x = None
        self.spline_y = None
        
    def fit(self, points, t):
        x = np.asarray(points[:, 0], dtype=float)
        y = np.asarray(points[:, 1], dtype=float)
        t_fit = np.asarray(t, dtype=float)

        # 对闭合曲线，确保首尾值完全一致
        if self.periodic:
            if (len(t_fit) == 0
                or not np.isclose(t_fit[-1], 1.0)
                or not np.isclose(x[0], x[-1])
                or not np.isclose(y[0], y[-1])):
                t_fit = np.concatenate([t_fit, [1.0]])
                x = np.concatenate([x, [x[0]]])
                y = np.concatenate([y, [y[0]]])
        
        if self.method == 'cubic_spline':
            bc_type = 'periodic' if self.periodic else 'not-a-knot'
            self.spline_x = CubicSpline(t_fit, x, bc_type=bc_type)
            self.spline_y = CubicSpline(t_fit, y, bc_type=bc_type)
            
        elif self.method == 'smooth_spline':
            s = self.smoothing if self.smoothing is not None else 0
            # 对于周期曲线，使用splprep（支持周期边界）
            if self.periodic:
                per = 1
                # splprep需要的数据格式
                data = [x, y]
                self.tck, _ = splprep(data, u=t_fit, s=s, k=3, per=per)
                self.spline_x = None  # 标记使用splprep
                self.spline_y = None
            else:
                # 非周期曲线使用UnivariateSpline
                self.spline_x = UnivariateSpline(t_fit, x, s=s, k=3)
                self.spline_y = UnivariateSpline(t_fit, y, s=s, k=3)
        else:
            raise ValueError(f"Unknown fitting method: {self.method}")
    
    def evaluate(self, t_eval):
        # 如果使用splprep（周期平滑样条）
        if hasattr(self, 'tck') and self.tck is not None:
            curve = splev(t_eval, self.tck)
            return np.column_stack(curve)
        # 否则使用UnivariateSpline或CubicSpline
        elif self.spline_x is not None and self.spline_y is not None:
            x_eval = self.spline_x(t_eval)
            y_eval = self.spline_y(t_eval)
            return np.column_stack([x_eval, y_eval])
        else:
            raise RuntimeError("Must call fit() before evaluate()")
    
    def get_derivatives(self, t_eval, order=1):
        dx = self.spline_x(t_eval, order)
        dy = self.spline_y(t_eval, order)
        return np.column_stack([dx, dy])




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_generator import generate_heart, add_noise
    from parameterization import parameterize_closed_curve
    
    points_clean, _ = generate_heart(n_points=30)
    points_noisy = add_noise(points_clean, noise_level=0.05)
    t = parameterize_closed_curve(points_noisy, method='chord')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    t_eval = np.linspace(0, 1, 500)
    
    fitter1 = ParametricCurveFitter(method='cubic_spline', periodic=True)
    fitter1.fit(points_noisy, t)
    curve1 = fitter1.evaluate(t_eval)
    axes[0].scatter(points_noisy[:, 0], points_noisy[:, 1], c='red', s=30, zorder=3)
    axes[0].plot(curve1[:, 0], curve1[:, 1], 'b-', linewidth=2)
    axes[0].plot(points_clean[:, 0], points_clean[:, 1], 'g--', alpha=0.5)
    axes[0].set_title('Cubic Spline Interpolation')
    axes[0].axis('equal'); axes[0].grid(True, alpha=0.3)
    
    fitter2 = ParametricCurveFitter(method='smooth_spline', smoothing=0.5, periodic=True)
    fitter2.fit(points_noisy, t)
    curve2 = fitter2.evaluate(t_eval)
    axes[1].scatter(points_noisy[:, 0], points_noisy[:, 1], c='red', s=30, zorder=3)
    axes[1].plot(curve2[:, 0], curve2[:, 1], 'b-', linewidth=2)
    axes[1].plot(points_clean[:, 0], points_clean[:, 1], 'g--', alpha=0.5)
    axes[1].set_title('Smooth Spline (Light)')
    axes[1].axis('equal'); axes[1].grid(True, alpha=0.3)
    
    fitter3 = ParametricCurveFitter(method='smooth_spline', smoothing=5.0, periodic=True)
    fitter3.fit(points_noisy, t)
    curve3 = fitter3.evaluate(t_eval)
    axes[2].scatter(points_noisy[:, 0], points_noisy[:, 1], c='red', s=30, zorder=3)
    axes[2].plot(curve3[:, 0], curve3[:, 1], 'b-', linewidth=2)
    axes[2].plot(points_clean[:, 0], points_clean[:, 1], 'g--', alpha=0.5)
    axes[2].set_title('Smooth Spline (Strong)')
    axes[2].axis('equal'); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_curve_fitting.png', dpi=150, bbox_inches='tight')
    print("Done")
