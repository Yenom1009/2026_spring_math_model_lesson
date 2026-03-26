import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io, color
from scipy.ndimage import convolve
from numba import njit
import threading
import os

# --- 1. 核心计算部分 (保持精度) ---
@njit
def find_seam_numba(energy, M, backtrack):
    r, c = energy.shape
    M[0, :] = energy[0, :]
    
    for i in range(1, r):
        for j in range(c):
            left = M[i-1, j-1] if j > 0 else np.inf
            mid = M[i-1, j]
            right = M[i-1, j+1] if j < c - 1 else np.inf
            m = min(left, min(mid, right))
            M[i, j] = energy[i, j] + m
            if m == left: backtrack[i, j] = j - 1
            elif m == mid: backtrack[i, j] = j
            else: backtrack[i, j] = j + 1
            
    seam = np.zeros(r, dtype=np.int32)
    seam[r-1] = np.argmin(M[r-1, :])
    for i in range(r-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam

def get_energy(im):
    gray = color.rgb2gray(im)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
    return convolve(gray, kernel) ** 2

def seam_carve_image(im, sz):
    target_h, target_w = sz
    working_im = im.copy()
    
    # 获取初始维度
    r, c, ch = working_im.shape
    
    # 宽度缩减逻辑
    while c > target_w:
        energy = get_energy(working_im)
        
        # 核心修改：针对当前 c 动态分配或切片
        # Numba 编译的函数要求传入的数组维度必须与函数内部操作一致
        # 我们在这里每次循环传入当前尺寸的视图
        M = np.zeros((r, c))
        backtrack = np.zeros((r, c), dtype=np.int32)
        
        # 将当前尺寸的能量矩阵和空数组传入
        seam = find_seam_numba(energy, M, backtrack)
        
        # 移除该接缝
        new_im = np.zeros((r, c - 1, ch), dtype=working_im.dtype)
        for i in range(r):
            # 这里的逻辑是：移除 seam[i] 这一列
            col_to_remove = seam[i]
            # 复制左侧和右侧的部分
            new_im[i, :col_to_remove, :] = working_im[i, :col_to_remove, :]
            new_im[i, col_to_remove:, :] = working_im[i, col_to_remove+1:, :]
            
        working_im = new_im
        c -= 1 # 尺寸更新
        
    return working_im

# --- 2. 界面与交互部分 ---
# 修复路径：请确保 bing1.png 和此脚本在同一目录下
img_path = '../figs/original.png'
if not os.path.exists(img_path):
    print(f"错误：找不到图片 {img_path}，请检查文件名！")
    exit()

im = io.imread(img_path)
if im.ndim == 3 and im.shape[2] == 4: im = im[:, :, :3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.25)
ax1.imshow(im); ax1.set_title('Input image'); ax1.axis('off')
himg = ax2.imshow(im); ax2.set_title('Resized Image'); ax2.axis('off')

slider_col = Slider(fig.add_axes([0.15, 0.10, 0.30, 0.03]), 'Col scale', 0.5, 1.0, valinit=1.0)
slider_row = Slider(fig.add_axes([0.15, 0.05, 0.30, 0.03]), 'Row scale', 0.5, 1.0, valinit=1.0)
btn = Button(fig.add_axes([0.60, 0.06, 0.20, 0.06]), 'Seam Carving', color='lightblue')

def on_click(event):
    btn.label.set_text("Computing...")
    def run():
        h, w = im.shape[:2]
        target_w = max(1, int(w * slider_col.val))
        target_h = max(1, int(h * slider_row.val))
        res = seam_carve_image(im, (target_h, target_w))
        himg.set_data(res)
        himg.set_extent([0, res.shape[1], res.shape[0], 0])
        ax2.set_title(f'Resized Image ({res.shape[0]}x{res.shape[1]})')
        fig.canvas.draw_idle()
        btn.label.set_text("Seam Carving")
    threading.Thread(target=run, daemon=True).start()

btn.on_clicked(on_click)
plt.show()