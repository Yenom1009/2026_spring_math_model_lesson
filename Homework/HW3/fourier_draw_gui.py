"""
傅里叶圆周运动合成 - 手绘GUI
用户在左侧画布上用鼠标绘制任意闭合曲线，
右侧动画展示圆盘链(epicycles)旋转合成该曲线的过程。

功能：
  - 手绘模式：鼠标在左侧画布自由绘制
  - 滑块调节傅里叶项数(1-100)
  - 播放/暂停动画
  - 清空重绘
  - 保存当前帧

数学原理：
  z(t) = sum_{k} c_k * exp(2*pi*i*k*t)
  c_k = (1/N) * sum_{j} z_j * exp(-2*pi*i*k*j/N)
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import math

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FourierDrawGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("傅里叶圆周运动合成 - 手绘版")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.drawing = False
        self.raw_points = []       # raw mouse points
        self.curve_points = []     # resampled points (complex)
        self.coeffs = []           # sorted (k, c_k) list
        self.n_terms = 20
        self.is_playing = False
        self.anim_t = 0.0
        self.speed = 1.0           # animation speed multiplier
        self.anim_step = 1.0 / 300  # 300 frames per cycle
        self.trace = []            # animation trace points
        self.save_counter = 0
        self.canvas_size = 500
        
        self.setup_ui()
    
    def setup_ui(self):
        # Top frame: two canvases side by side
        top_frame = tk.Frame(self.root, bg='#1a1a2e')
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left canvas: drawing area
        left_frame = tk.Frame(top_frame, bg='#1a1a2e')
        left_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(left_frame, text="✏️ 绘制区（按住鼠标画闭合曲线）", 
                 bg='#1a1a2e', fg='white', font=('Microsoft YaHei', 11, 'bold')
                ).pack(pady=2)
        
        self.draw_canvas = tk.Canvas(left_frame, width=self.canvas_size, 
                                      height=self.canvas_size, bg='white',
                                      cursor='crosshair', highlightthickness=2,
                                      highlightbackground='#e94560')
        self.draw_canvas.pack()
        
        self.draw_canvas.bind('<ButtonPress-1>', self.on_press)
        self.draw_canvas.bind('<B1-Motion>', self.on_drag)
        self.draw_canvas.bind('<ButtonRelease-1>', self.on_release)
        
        # Right canvas: animation area
        right_frame = tk.Frame(top_frame, bg='#1a1a2e')
        right_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(right_frame, text=" 傅里叶圆盘链动画", 
                 bg='#1a1a2e', fg='white', font=('Microsoft YaHei', 11, 'bold')
                ).pack(pady=2)
        
        self.anim_canvas = tk.Canvas(right_frame, width=self.canvas_size,
                                      height=self.canvas_size, bg='#f8f8f8',
                                      highlightthickness=2,
                                      highlightbackground='#4CAF50')
        self.anim_canvas.pack()
        
        # Bottom controls
        ctrl_frame = tk.Frame(self.root, bg='#1a1a2e')
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Slider for n_terms
        tk.Label(ctrl_frame, text="傅里叶项数 N:", bg='#1a1a2e', fg='white',
                 font=('Microsoft YaHei', 10)).pack(side=tk.LEFT, padx=5)
        
        self.slider_var = tk.IntVar(value=self.n_terms)
        self.slider = ttk.Scale(ctrl_frame, from_=1, to=100, 
                                variable=self.slider_var, orient=tk.HORIZONTAL,
                                length=300, command=self.on_slider_change)
        self.slider.pack(side=tk.LEFT, padx=5)
        
        self.label_n = tk.Label(ctrl_frame, text=f"N={self.n_terms}", 
                                bg='#1a1a2e', fg='#e94560',
                                font=('Microsoft YaHei', 11, 'bold'))
        self.label_n.pack(side=tk.LEFT, padx=5)
        
        # Speed slider
        tk.Label(ctrl_frame, text="速度:", bg='#1a1a2e', fg='white',
                 font=('Microsoft YaHei', 10)).pack(side=tk.LEFT, padx=(15, 5))
        
        self.speed_var = tk.DoubleVar(value=self.speed)
        self.speed_slider = ttk.Scale(ctrl_frame, from_=0.1, to=3.0, 
                                      variable=self.speed_var, orient=tk.HORIZONTAL,
                                      length=150, command=self.on_speed_change)
        self.speed_slider.pack(side=tk.LEFT, padx=5)
        
        self.label_speed = tk.Label(ctrl_frame, text=f"{self.speed:.1f}x", 
                                    bg='#1a1a2e', fg='#4CAF50',
                                    font=('Microsoft YaHei', 10, 'bold'))
        self.label_speed.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_style = {'font': ('Microsoft YaHei', 10), 'width': 8}
        
        self.btn_play = tk.Button(ctrl_frame, text="▶ 播放", bg='#4CAF50', fg='white',
                                   command=self.toggle_play, **btn_style)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl_frame, text="🗑 清空", bg='#e94560', fg='white',
                  command=self.clear_all, **btn_style).pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl_frame, text="💾 保存", bg='#2196F3', fg='white',
                  command=self.save_frame, **btn_style).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="请在左侧画布上绘制一个闭合曲线，然后点击播放")
        tk.Label(self.root, textvariable=self.status_var, bg='#1a1a2e', fg='#aaa',
                 font=('Microsoft YaHei', 9)).pack(pady=2)
    
    # ==================== Drawing ====================
    
    def on_press(self, event):
        self.drawing = True
        self.raw_points = [(event.x, event.y)]
        self.is_playing = False
        self.btn_play.config(text=" 播放")
        self.draw_canvas.delete('all')
        self.anim_canvas.delete('all')
        self.trace = []
        self.status_var.set("正在绘制...松开鼠标完成")
    
    def on_drag(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        # Clamp to canvas
        x = max(5, min(self.canvas_size - 5, x))
        y = max(5, min(self.canvas_size - 5, y))
        self.raw_points.append((x, y))
        
        if len(self.raw_points) >= 2:
            x0, y0 = self.raw_points[-2]
            self.draw_canvas.create_line(x0, y0, x, y, fill='#333', width=2.5)
    
    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        
        if len(self.raw_points) < 10:
            self.status_var.set("绘制的点太少，请重新绘制")
            return
        
        # Close the curve
        self.raw_points.append(self.raw_points[0])
        x0, y0 = self.raw_points[-2]
        x1, y1 = self.raw_points[-1]
        self.draw_canvas.create_line(x0, y0, x1, y1, fill='#333', width=2.5)
        
        # Process: resample to uniform N points
        self._process_curve()
        self.status_var.set(f"绘制完成！{len(self.curve_points)}个采样点，"
                           f"{len(self.coeffs)}个傅里叶项。点击 播放 开始动画。")
    
    def _process_curve(self):
        """Resample raw points to N uniform points, compute DFT"""
        pts = np.array(self.raw_points, dtype=float)
        
        # Center and normalize to [-1, 1]
        cx, cy = self.canvas_size / 2, self.canvas_size / 2
        pts[:, 0] = (pts[:, 0] - cx) / (self.canvas_size / 2)
        pts[:, 1] = -(pts[:, 1] - cy) / (self.canvas_size / 2)  # flip y
        
        # Compute cumulative arc length
        diffs = np.diff(pts, axis=0)
        seg_len = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        cum_len = np.concatenate([[0], np.cumsum(seg_len)])
        total_len = cum_len[-1]
        
        if total_len < 1e-6:
            return
        
        cum_len /= total_len  # normalize to [0, 1]
        
        # Resample to 500 uniform points
        N = 500
        t_uniform = np.linspace(0, 1, N, endpoint=False)
        x_resamp = np.interp(t_uniform, cum_len, pts[:, 0])
        y_resamp = np.interp(t_uniform, cum_len, pts[:, 1])
        
        self.curve_points = x_resamp + 1j * y_resamp
        self._compute_dft()
    
    def _compute_dft(self):
        """Compute DFT coefficients and sort by amplitude"""
        z = self.curve_points
        N = len(z)
        coeffs = {}
        
        for k in range(-(N // 2), N // 2 + 1):
            c_k = np.sum(z * np.exp(-2j * np.pi * k * np.arange(N) / N)) / N
            coeffs[k] = c_k
        
        # Sort by amplitude (descending)
        sorted_keys = sorted(coeffs.keys(), key=lambda k: abs(coeffs[k]), reverse=True)
        self.all_coeffs = [(k, coeffs[k]) for k in sorted_keys]
        self._update_terms()
    
    def _update_terms(self):
        """Select top n_terms coefficients, ensuring k=0 is first (fixed center)"""
        # Separate DC component (k=0) from others
        dc = [(k, c) for k, c in self.all_coeffs if k == 0]
        others = [(k, c) for k, c in self.all_coeffs if k != 0]
        # DC first, then top (n_terms-1) by amplitude
        self.coeffs = dc + others[:max(0, self.n_terms - len(dc))]
    
    # ==================== Animation ====================
    
    def toggle_play(self):
        if len(self.curve_points) == 0:
            self.status_var.set("请先绘制曲线！")
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ 暂停")
            self.trace = []
            self.anim_t = 0.0
            self._animate()
        else:
            self.btn_play.config(text=" 播放")
    
    def _animate(self):
        if not self.is_playing:
            return
        
        self._draw_frame(self.anim_t)
        self.anim_t += self.anim_step * self.speed
        
        if self.anim_t >= 1.0:
            self.anim_t -= 1.0
            self.trace = []  # clear trace for new cycle
        
        self.root.after(16, self._animate)  # ~60fps
    
    def _draw_frame(self, t):
        self.anim_canvas.delete('all')
        
        cs = self.canvas_size
        cx, cy = cs / 2, cs / 2
        scale = cs / 2.5  # leave margin
        
        # Draw original curve (gray)
        if len(self.curve_points) > 1:
            orig_coords = []
            for z in self.curve_points:
                px = cx + z.real * scale
                py = cy - z.imag * scale
                orig_coords.extend([px, py])
            # Close it
            orig_coords.extend(orig_coords[:2])
            self.anim_canvas.create_line(*orig_coords, fill='#ccc', width=1.5,
                                          smooth=False)
        
        # Draw epicycle chain
        x, y = 0.0, 0.0
        
        for k, c_k in self.coeffs:
            radius = abs(c_k)
            if radius < 1e-8:
                continue
            
            # Screen coordinates of circle center
            scr_x = cx + x * scale
            scr_y = cy - y * scale
            scr_r = radius * scale
            
            # Draw circle
            if scr_r > 1:
                self.anim_canvas.create_oval(scr_x - scr_r, scr_y - scr_r,
                                              scr_x + scr_r, scr_y + scr_r,
                                              outline='#bbb', width=1)
            
            # Next point
            if k == 0:
                # DC component: no rotation, just offset
                angle = math.atan2(c_k.imag, c_k.real)
            else:
                # Rotating component
                angle = 2 * math.pi * k * t + math.atan2(c_k.imag, c_k.real)
            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)
            
            # Draw connecting line (arm)
            new_scr_x = cx + new_x * scale
            new_scr_y = cy - new_y * scale
            self.anim_canvas.create_line(scr_x, scr_y, new_scr_x, new_scr_y,
                                          fill='#555', width=1.2)
            
            x, y = new_x, new_y
        
        # Pen tip position
        tip_scr_x = cx + x * scale
        tip_scr_y = cy - y * scale
        
        # Add to trace
        self.trace.append((tip_scr_x, tip_scr_y))
        
        # Limit trace length
        max_trace = int(1.0 / self.anim_step) + 10
        if len(self.trace) > max_trace:
            self.trace = self.trace[-max_trace:]
        
        # Draw trace (red curve)
        if len(self.trace) >= 2:
            coords = []
            for px, py in self.trace:
                coords.extend([px, py])
            self.anim_canvas.create_line(*coords, fill='#e94560', width=2.5,
                                          smooth=True)
        
        # Draw pen tip (green dot)
        r = 5
        self.anim_canvas.create_oval(tip_scr_x - r, tip_scr_y - r,
                                      tip_scr_x + r, tip_scr_y + r,
                                      fill='#00ff88', outline='#00cc66')
        
        # Info text
        self.anim_canvas.create_text(10, 15, anchor='nw', 
                                      text=f"N = {self.n_terms}  |  t = {t:.3f}",
                                      font=('Consolas', 11), fill='#333')
        
        progress = min(100, int(t * 100))
        self.status_var.set(f"动画播放中... 傅里叶项数 N={self.n_terms}  进度: {progress}%")
    
    # ==================== Controls ====================
    
    def on_slider_change(self, val):
        self.n_terms = int(float(val))
        self.label_n.config(text=f"N={self.n_terms}")
        if hasattr(self, 'all_coeffs') and len(self.all_coeffs) > 0:
            self._update_terms()
            self.trace = []
            self.anim_t = 0.0
    
    def on_speed_change(self, val):
        self.speed = float(val)
        self.label_speed.config(text=f"{self.speed:.1f}x")
    
    def clear_all(self):
        self.is_playing = False
        self.btn_play.config(text=" 播放")
        self.draw_canvas.delete('all')
        self.anim_canvas.delete('all')
        self.raw_points = []
        self.curve_points = []
        self.coeffs = []
        self.trace = []
        self.anim_t = 0.0
        self.status_var.set("已清空。请在左侧重新绘制曲线。")
    
    def save_frame(self):
        """Save both canvases as PostScript then hint user"""
        self.save_counter += 1
        # Save animation canvas
        fn = f'fourier_draw_frame_{self.save_counter}.ps'
        fp = os.path.join(OUTPUT_DIR, fn)
        self.anim_canvas.postscript(file=fp, colormode='color')
        self.status_var.set(f"已保存: {fp}")
        print(f"Saved: {fp}")


# ==================== Main ====================

if __name__ == '__main__':
    root = tk.Tk()
    # Set window size
    root.geometry("1060x620")
    root.resizable(False, False)
    app = FourierDrawGUI(root)
    root.mainloop()
