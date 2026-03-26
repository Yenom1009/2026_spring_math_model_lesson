import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io
from scipy.ndimage import convolve

## read image
im = io.imread('../figs/original.png')
if im.ndim == 3 and im.shape[2] == 4:
    im = im[:, :, :3]

## draw 2 copies of the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.22)
ax1.imshow(im)
ax1.set_title('Input image')
ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im))
ax2.set_title('Resized Image\nAdjust sliders and click the button')
ax2.axis('off')

slider_col_ax = fig.add_axes([0.15, 0.10, 0.30, 0.03])
slider_row_ax = fig.add_axes([0.15, 0.05, 0.30, 0.03])
slider_col = Slider(slider_col_ax, 'Col scale', 0.5, 2.0, valinit=1.0)
slider_row = Slider(slider_row_ax, 'Row scale', 0.5, 2.0, valinit=1.0)

btn_ax = fig.add_axes([0.60, 0.06, 0.20, 0.06])
btn = Button(btn_ax, 'Seam Carving', color='lightblue', hovercolor='deepskyblue')

def on_click(event):
    h, w = im.shape[:2]
    target_w = max(1, int(w * slider_col.val))
    target_h = max(1, int(h * slider_row.val))
    result = seam_carve_image(im, (target_h, target_w))
    himg.set_data(result)
    himg.set_extent([0, result.shape[1], result.shape[0], 0])
    ax2.set_title(f'Resized Image ({result.shape[0]}x{result.shape[1]})')
    fig.canvas.draw_idle()

btn.on_clicked(on_click)


# ## TODO: implement function: seam_carve_image
# def seam_carve_image(im, sz):
#     """Seam carving to resize image to target size.

#     Args:
#         im: (h, w, 3) input RGB image (uint8)
#         sz: (target_h, target_w) target size

#     Returns:
#         resized image of shape (target_h, target_w, 3)
#     """
#     raise NotImplementedError

# def get_energy(im):
#     """严格按照实验要求：计算RGB三通道拉普拉斯算子的平方和"""
#     # 定义拉普拉斯核
#     kernel = np.array([[0.5, 1.0, 0.5], 
#                        [1.0, -6.0, 1.0],[0.5, 1.0, 0.5]])
    
#     h, w, _ = im.shape
#     energy = np.zeros((h, w), dtype=np.float32)
    
#     # 遍历 R, G, B 三个通道
#     for c in range(3):
#         channel = im[:, :, c].astype(np.float32)
#         # 对单个通道进行拉普拉斯滤波
#         laplacian = convolve(channel, kernel, mode='reflect')
#         # 平方并累加到总能量图中
#         energy += laplacian ** 2
        
#     return energy

def get_energy_e1(im):
    """
    论文原版的 e1 能量函数 (一阶梯度幅值的 L1 范数)
    能够更好地保护物体的结构边缘，而非被水波噪声干扰
    """
    # 转换为灰度图计算梯度
    gray = np.mean(im.astype(np.float32), axis=2)
    # 计算 x 和 y 方向的一阶梯度
    dy, dx = np.gradient(gray)
    # 计算绝对值之和
    energy = np.abs(dx) + np.abs(dy)
    return energy

def get_vertical_seam(energy):
    """利用 NumPy 向量化寻找最优路径 (修复平滑区域偏移Bug版)"""
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=int)
    
    for i in range(1, h):
        m_mid = M[i-1, :]
        m_left = np.roll(m_mid, 1)
        m_right = np.roll(m_mid, -1)
        
        # 处理边界，防止跨界绕回
        m_left[0] = np.inf
        m_right[-1] = np.inf
        
        # 核心修复点：将 m_mid 放在第一位 [m_mid, m_left, m_right]
        # 这样遇到平滑区域（三者相等）时，默认选择直走(m_mid)
        choices = np.stack([m_mid, m_left, m_right])
        idx = np.argmin(choices, axis=0)
        
        # 将索引映射回真实的水平偏移量
        shift = np.zeros(w, dtype=int)
        shift[idx == 1] = -1  # 索引1是m_left，往左偏
        shift[idx == 2] = 1   # 索引2是m_right，往右偏
        
        backtrack[i, :] = np.arange(w) + shift
        M[i, :] += np.min(choices, axis=0)
            
    # 回溯寻找路径
    seam = np.zeros(h, dtype=int)
    seam[h-1] = np.argmin(M[h-1, :])
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam

def remove_vertical_seam(im, seam):
    """利用 NumPy 索引高效移除像素"""
    h, w, c = im.shape
    # 创建掩码
    mask = np.ones((h, w), dtype=bool)
    for i in range(h):
        mask[i, seam[i]] = False
    
    # 重塑数组，移除被标记的像素
    return im[mask].reshape(h, w - 1, c)

# def seam_carve_image(im, sz):
#     """主函数，处理多步缩减"""
#     target_h, target_w = sz
#     curr_im = im.copy()
    
#     # 缩减宽度
#     while curr_im.shape[1] > target_w:
#         energy = get_energy_e1(curr_im)
#         seam = get_vertical_seam(energy)
#         curr_im = remove_vertical_seam(curr_im, seam)
        
#     # 缩减高度 (通过转置复用代码)
#     while curr_im.shape[0] > target_h:
#         curr_im = np.transpose(curr_im, (1, 0, 2))
#         energy = get_energy_e1(curr_im)
#         seam = get_vertical_seam(energy)
#         curr_im = remove_vertical_seam(curr_im, seam)
#         curr_im = np.transpose(curr_im, (1, 0, 2))
        
#     return curr_im

def seam_carve_image(im, sz):
    """主函数：交替缩减宽度和高度，防止单一维度过度压缩"""
    target_h, target_w = sz
    curr_im = im.copy()
    
    # 只要宽或高还没达到目标，就继续循环
    while curr_im.shape[1] > target_w or curr_im.shape[0] > target_h:
        
        # 缩减 1 像素宽度
        if curr_im.shape[1] > target_w:
            # 这里调用上面提供的 get_energy_e1 效果更好
            energy = get_energy_e1(curr_im) 
            seam = get_vertical_seam(energy)
            curr_im = remove_vertical_seam(curr_im, seam)
            
        # 缩减 1 像素高度
        if curr_im.shape[0] > target_h:
            # 矩阵转置，将高度问题转化为宽度问题处理
            curr_im = np.transpose(curr_im, (1, 0, 2))
            energy = get_energy_e1(curr_im)
            seam = get_vertical_seam(energy)
            curr_im = remove_vertical_seam(curr_im, seam)
            # 转置回来
            curr_im = np.transpose(curr_im, (1, 0, 2))
            
    return curr_im

plt.show()