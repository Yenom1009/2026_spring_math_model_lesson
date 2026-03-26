import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io
from skimage.transform import resize
from scipy.ndimage import convolve

# ----------------- 算法核心部分 -----------------

# def get_energy_e1(im):
#     """一阶梯度能量函数 (论文原版最优方案)"""
#     gray = np.mean(im.astype(np.float32), axis=2)
#     dy, dx = np.gradient(gray)
#     return np.abs(dx) + np.abs(dy)

def get_energy(im):
    """
    实验要求的二阶能量函数：计算 RGB 三通道拉普拉斯算子的平方和
    """
    # 定义实验指导书中要求的 Laplacian 滤波核
    kernel = np.array([[0.5, 1.0, 0.5], 
                       [1.0, -6.0, 1.0], 
                       [0.5, 1.0, 0.5]])
    
    h, w, _ = im.shape
    energy = np.zeros((h, w), dtype=np.float32)
    
    # 遍历 R, G, B 三个通道
    for c in range(3):
        # 提取单通道并转换为浮点型以防止溢出
        channel = im[:, :, c].astype(np.float32)
        
        # 对该通道进行拉普拉斯滤波
        laplacian = convolve(channel, kernel, mode='reflect')
        
        # 将该通道的拉普拉斯结果求平方，并累加到总能量图中
        energy += laplacian ** 2
        
    return energy

def get_vertical_seam(energy):
    """向量化动态规划寻找垂直最优接缝"""
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=int)
    
    for i in range(1, h):
        m_mid = M[i-1, :]
        m_left = np.roll(m_mid, 1)
        m_right = np.roll(m_mid, -1)
        
        m_left[0] = np.inf
        m_right[-1] = np.inf
        
        # m_mid 放在首位，防止平滑区域产生对角线拉扯偏移
        choices = np.stack([m_mid, m_left, m_right])
        idx = np.argmin(choices, axis=0)
        
        shift = np.zeros(w, dtype=int)
        shift[idx == 1] = -1
        shift[idx == 2] = 1
        
        backtrack[i, :] = np.arange(w) + shift
        M[i, :] += np.min(choices, axis=0)
            
    seam = np.zeros(h, dtype=int)
    seam[h-1] = np.argmin(M[h-1, :])
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam

def remove_vertical_seam(im, seam):
    """缩小图像：移除接缝"""
    h, w, c = im.shape
    mask = np.ones((h, w), dtype=bool)
    for i in range(h):
        mask[i, seam[i]] = False
    return im[mask].reshape(h, w - 1, c)

def insert_vertical_seam_with_penalty(im, seam, penalty):
    """放大图像：插入接缝并更新惩罚矩阵"""
    h, w, c = im.shape
    new_im = np.zeros((h, w + 1, c), dtype=im.dtype)
    new_penalty = np.zeros((h, w + 1), dtype=np.float32)
    
    for i in range(h):
        col = seam[i]
        # 对相邻像素取平均值进行插值
        p1 = im[i, col, :].astype(np.float32)
        p2 = im[i, min(col+1, w-1), :].astype(np.float32)
        new_pixel = ((p1 + p2) / 2).astype(im.dtype)
        
        # 拼装新图像
        new_im[i, :col+1, :] = im[i, :col+1, :]
        new_im[i, col+1, :] = new_pixel
        new_im[i, col+2:, :] = im[i, col+1:, :]
        
        # 拼装新惩罚矩阵，并将惩罚平移
        new_penalty[i, :col+1] = penalty[i, :col+1]
        new_penalty[i, col+2:] = penalty[i, col+1:]
        
        # 【防拉伸优化】：不仅惩罚新插入的像素，还略微惩罚左右相邻像素
        # 强制算法下次去寻找其他区域，从而使放大的部分均匀分布在原图中
        new_penalty[i, max(0, col):min(w+1, col+3)] = 1e6  
        
    return new_im, new_penalty


def seam_carve_image(im, sz):
    """主函数：为了防止变量污染，我们将宽和高的缩放独立为清晰的步骤"""
    target_h, target_w = sz
    curr_im = im.copy()
    
    print(f"开始处理: 当前尺寸 {curr_im.shape[:2]} -> 目标尺寸 {target_h}x{target_w}")
    
    # --- 1. 专门处理宽度 (列) ---
    if curr_im.shape[1] > target_w: # 缩小
        while curr_im.shape[1] > target_w:
            energy = get_energy(curr_im)
            seam = get_vertical_seam(energy)
            curr_im = remove_vertical_seam(curr_im, seam)
            
    elif curr_im.shape[1] < target_w: # 放大
        # 【核心修复】：惩罚矩阵必须在循环外初始化，让它有记忆！
        penalty = np.zeros((curr_im.shape[0], curr_im.shape[1]), dtype=np.float32)
        while curr_im.shape[1] < target_w:
            energy = get_energy(curr_im) + penalty
            seam = get_vertical_seam(energy)
            curr_im, penalty = insert_vertical_seam_with_penalty(curr_im, seam, penalty)
            
    # --- 2. 专门处理高度 (行) ---
    if curr_im.shape[0] > target_h: # 缩小
        curr_im = np.transpose(curr_im, (1, 0, 2))
        while curr_im.shape[1] > target_h:  # 注意转置后，高度变成了 shape[1]
            energy = get_energy(curr_im)
            seam = get_vertical_seam(energy)
            curr_im = remove_vertical_seam(curr_im, seam)
        curr_im = np.transpose(curr_im, (1, 0, 2))
        
    elif curr_im.shape[0] < target_h: # 放大
        curr_im = np.transpose(curr_im, (1, 0, 2))
        penalty = np.zeros((curr_im.shape[0], curr_im.shape[1]), dtype=np.float32)
        while curr_im.shape[1] < target_h:
            energy = get_energy(curr_im) + penalty
            seam = get_vertical_seam(energy)
            curr_im, penalty = insert_vertical_seam_with_penalty(curr_im, seam, penalty)
        curr_im = np.transpose(curr_im, (1, 0, 2))
        
    print("处理完成！")
    return curr_im

def amplify_content(im, scale_factor=1.2):
    """内容放大 (Content Amplification)"""
    h, w = im.shape[:2]
    target_h, target_w = int(h * scale_factor), int(w * scale_factor)
    
    print(f"内容放大步骤 1: 传统插值放大到 {target_h}x{target_w}")
    # preserve_range=True 和 astype(uint8) 保证颜色不失真
    enlarged_im = resize(im, (target_h, target_w), preserve_range=True).astype(np.uint8)
    
    print(f"内容放大步骤 2: Seam Carving 裁切回原尺寸 {h}x{w}")
    final_im = seam_carve_image(enlarged_im, (h, w))
    return final_im


# ----------------- GUI 界面部分 -----------------

## 读取图像
try:
    im = io.imread('../figs/original.png') # 请确保路径正确
except FileNotFoundError:
    # 容错：如果找不到文件，生成一个测试渐变图
    print("未找到图片，生成测试图像...")
    im = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

if im.ndim == 3 and im.shape[2] == 4:
    im = im[:, :, :3]

## 绘制UI
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.subplots_adjust(bottom=0.25)
ax1.imshow(im)
ax1.set_title('Input image')
ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im))
ax2.set_title('Result Image\nAdjust sliders or click Amplification')
ax2.axis('off')

slider_col_ax = fig.add_axes([0.15, 0.12, 0.30, 0.03])
slider_row_ax = fig.add_axes([0.15, 0.06, 0.30, 0.03])
# 缩放范围设置为 0.5 到 1.5，允许放大和缩小
slider_col = Slider(slider_col_ax, 'Col scale', 0.5, 1.5, valinit=1.0)
slider_row = Slider(slider_row_ax, 'Row scale', 0.5, 1.5, valinit=1.0)

# 添加两个按钮
btn_resize_ax = fig.add_axes([0.55, 0.08, 0.15, 0.06])
btn_resize = Button(btn_resize_ax, 'Seam Carving', color='lightblue', hovercolor='deepskyblue')

btn_amp_ax = fig.add_axes([0.75, 0.08, 0.18, 0.06])
btn_amp = Button(btn_amp_ax, 'Content Amplify(1.2x)', color='lightgreen', hovercolor='limegreen')

def update_display(result, title):
    himg.set_data(result)
    himg.set_extent([0, result.shape[1], result.shape[0], 0])
    ax2.set_title(title)
    fig.canvas.draw_idle()

def on_click_resize(event):
    h, w = im.shape[:2]
    target_w = max(1, int(w * slider_col.val))
    target_h = max(1, int(h * slider_row.val))
    result = seam_carve_image(im, (target_h, target_w))
    update_display(result, f'Resized Image ({result.shape[0]}x{result.shape[1]})')

def on_click_amplify(event):
    # 内容放大功能直接以 1.2 倍进行演示
    result = amplify_content(im, scale_factor=1.2)
    update_display(result, f'Amplified Content ({result.shape[0]}x{result.shape[1]})')

btn_resize.on_clicked(on_click_resize)
btn_amp.on_clicked(on_click_amplify)

plt.show()