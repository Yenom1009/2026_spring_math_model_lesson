import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RectangleSelector
from skimage import io
from skimage.transform import resize as sk_resize
from scipy.ndimage import convolve
import matplotlib.patches as patches

# ----------------- 1. 核心算法：能量计算 -----------------

# def get_energy(im, mask=None):
#     """
#     实验要求：二阶拉普拉斯算子平方和。
#     支持 Mask 负能量引导（用于物体移除）。
#     """
#     kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
#     h, w, _ = im.shape
#     energy = np.zeros((h, w), dtype=np.float32)
    
#     # 分通道计算能量
#     for c in range(3):
#         channel = im[:, :, c].astype(np.float32)
#         laplacian = convolve(channel, kernel, mode='reflect')
#         energy += laplacian ** 2
        
#     # 物体移除模式：选区强制为极小能量（黑洞吸引）
#     if mask is not None:
#         energy[mask] = -1e8 
        
#     return energy

def get_energy(im, mask=None):
    """
    一阶能量函数 (e1): 基于梯度幅值的 L1 范数。
    相比二阶算子，它对图像主体结构的保护更稳健，不易被细小噪声干扰。
    """
    h, w, _ = im.shape
    energy = np.zeros((h, w), dtype=np.float32)
    
    # 分通道计算一阶梯度
    for c in range(3):
        channel = im[:, :, c].astype(np.float32)
        # 使用 np.gradient 计算 x 和 y 方向的偏导数
        dy, dx = np.gradient(channel)
        # 累加一阶梯度幅值：|dx| + |dy|
        energy += (np.abs(dx) + np.abs(dy))
        
    # 物体移除模式：选区强制为极小能量（黑洞吸引）
    if mask is not None:
        energy[mask] = -1e8 
        
    return energy

# ----------------- 2. 核心算法：寻路 (DP) -----------------

def get_vertical_seam(energy):
    """
    动态规划寻找最优接缝。
    使用向量化提速，并修复了平滑区域的偏移 Bug。
    """
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=int)
    
    for i in range(1, h):
        m_mid = M[i-1, :]
        m_left = np.roll(m_mid, 1)
        m_right = np.roll(m_mid, -1)
        
        m_left[0] = np.inf
        m_right[-1] = np.inf
        
        # 优先级：中 > 左 > 右 (防止平滑区产生侧移)
        choices = np.stack([m_mid, m_left, m_right])
        idx = np.argmin(choices, axis=0)
        
        shift = np.zeros(w, dtype=int)
        shift[idx == 1] = -1
        shift[idx == 2] = 1
        
        backtrack[i, :] = (np.arange(w) + shift).astype(int)
        M[i, :] += np.min(choices, axis=0)
            
    seam = np.zeros(h, dtype=int)
    seam[h-1] = np.argmin(M[h-1, :])
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, int(seam[i+1])]
    return seam.astype(int)

# ----------------- 3. 核心算法：图像操作 -----------------

def remove_vertical_seam(im, seam, mask=None):
    """移除接缝：物理删除像素点，防止索引错位"""
    h, w, c = im.shape
    new_im = np.zeros((h, w - 1, c), dtype=im.dtype)
    new_mask = np.zeros((h, w - 1), dtype=bool) if mask is not None else None
    
    for i in range(h):
        col = seam[i]
        new_im[i, :, :] = np.delete(im[i, :, :], col, axis=0)
        if mask is not None:
            new_mask[i, :] = np.delete(mask[i, :], col, axis=0)
            
    return (new_im, new_mask) if mask is not None else new_im

def insert_vertical_seam(im, seam, penalty):
    """插入接缝：通过平均插值生成新像素，并施加惩罚以分散接缝"""
    h, w, c = im.shape
    new_im = np.zeros((h, w + 1, c), dtype=im.dtype)
    new_penalty = np.zeros((h, w + 1), dtype=np.float32)
    
    for i in range(h):
        col = seam[i]
        # 插值生成新像素
        p1 = im[i, col, :].astype(float)
        p2 = im[i, min(col+1, w-1), :].astype(float)
        new_pixel = ((p1 + p2) / 2).astype(im.dtype)
        
        new_im[i, :col+1, :] = im[i, :col+1, :]
        new_im[i, col+1, :] = new_pixel
        new_im[i, col+2:, :] = im[i, col+1:, :]
        
        new_penalty[i, :col+1] = penalty[i, :col+1]
        new_penalty[i, col+2:] = penalty[i, col+1:]
        # 插入惩罚，强制下次分散
        new_penalty[i, max(0, col):min(w+1, col+3)] = 1e7
        
    return new_im, new_penalty

# ----------------- 4. 应用层：移除、放大、缩放 -----------------

def object_removal(im, mask):
    """
    论文标准物体移除：
    1. 判定最短方向。
    2. 移除直到物体消失。
    3. 插入等量接缝恢复原图尺寸。
    """
    curr_im = im.copy()
    curr_mask = mask.copy()
    rows, cols = np.where(curr_mask)
    if len(rows) == 0: return curr_im
    
    # 论文 4.6 节：选择短轴移除
    is_hor = (np.max(rows)-np.min(rows)) < (np.max(cols)-np.min(cols))
    if is_hor:
        curr_im = np.transpose(curr_im, (1, 0, 2))
        curr_mask = np.transpose(curr_mask, (1, 0))

    count = 0
    # 阶段 1: 移除
    while np.any(curr_mask) and curr_im.shape[1] > 2:
        energy = get_energy(curr_im, curr_mask)
        seam = get_vertical_seam(energy)
        curr_im, curr_mask = remove_vertical_seam(curr_im, seam, curr_mask)
        count += 1
    
    # 阶段 2: 恢复尺寸
    print(f"物体已消失，补回 {count} 条接缝...")
    penalty = np.zeros((curr_im.shape[0], curr_im.shape[1]), dtype=np.float32)
    for _ in range(count):
        energy = get_energy(curr_im) + penalty
        seam = get_vertical_seam(energy)
        curr_im, penalty = insert_vertical_seam(curr_im, seam, penalty)

    if is_hor:
        curr_im = np.transpose(curr_im, (1, 0, 2))
    return curr_im

def seam_carve_image(im, sz):
    """内容感知缩放（支持放大缩小）"""
    th, tw = sz
    curr = im.copy()
    
    # 宽度处理
    if curr.shape[1] > tw:
        while curr.shape[1] > tw:
            curr = remove_vertical_seam(curr, get_vertical_seam(get_energy(curr)))
    elif curr.shape[1] < tw:
        p = np.zeros((curr.shape[0], curr.shape[1]))
        while curr.shape[1] < tw:
            curr, p = insert_vertical_seam(curr, get_vertical_seam(get_energy(curr)+p), p)
            
    # 高度处理
    if curr.shape[0] != th:
        curr = np.transpose(curr, (1, 0, 2))
        if curr.shape[1] > th:
            while curr.shape[1] > th:
                curr = remove_vertical_seam(curr, get_vertical_seam(get_energy(curr)))
        elif curr.shape[1] < th:
            p = np.zeros((curr.shape[0], curr.shape[1]))
            while curr.shape[1] < th:
                curr, p = insert_vertical_seam(curr, get_vertical_seam(get_energy(curr)+p), p)
        curr = np.transpose(curr, (1, 0, 2))
    return curr

def amplify_content(im):
    """内容放大：1.2倍插值放大后再裁切回原尺寸"""
    h, w = im.shape[:2]
    enlarged = sk_resize(im, (int(h*1.2), int(w*1.2)), preserve_range=True).astype(np.uint8)
    return seam_carve_image(enlarged, (h, w))

# ----------------- 5. GUI 交互逻辑 -----------------

try:
    im = io.imread('../figs/original.png')
except:
    im = np.random.randint(0, 255, (300, 450, 3), dtype=np.uint8)

if im.ndim == 3 and im.shape[2] == 4: im = im[:, :, :3]
current_mask = np.zeros((im.shape[0], im.shape[1]), dtype=bool)
rect_patch = None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(bottom=0.25)
ax1.imshow(im); ax1.set_title('Input: Draw Box to Remove Object'); ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im)); ax2.set_title('Result'); ax2.axis('off')

def onselect(eclick, erelease):
    global current_mask, rect_patch
    if rect_patch: 
        try: rect_patch.remove()
        except: pass
    x1, y1, x2, y2 = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
    xmin, xmax = max(0, min(x1, x2)), min(im.shape[1], max(x1, x2))
    ymin, ymax = max(0, min(y1, y2)), min(im.shape[0], max(y1, y2))
    current_mask.fill(False); current_mask[ymin:ymax, xmin:xmax] = True
    rect_patch = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='red', alpha=0.3)
    ax1.add_patch(rect_patch); fig.canvas.draw_idle()

selector = RectangleSelector(ax1, onselect, interactive=True)
s_col = Slider(fig.add_axes([0.1, 0.12, 0.25, 0.03]), 'Col Scale', 0.5, 1.5, valinit=1.0)
s_row = Slider(fig.add_axes([0.1, 0.07, 0.25, 0.03]), 'Row Scale', 0.5, 1.5, valinit=1.0)

def update_ui(res, msg):
    himg.set_data(res); himg.set_extent([0, res.shape[1], res.shape[0], 0])
    ax2.set_title(msg); fig.canvas.draw_idle()

btn_res = Button(fig.add_axes([0.4, 0.08, 0.12, 0.06]), 'Resize', color='lightblue')
btn_res.on_clicked(lambda e: update_ui(seam_carve_image(im, (int(im.shape[0]*s_row.val), int(im.shape[1]*s_col.val))), 'Resized'))

btn_amp = Button(fig.add_axes([0.55, 0.08, 0.15, 0.06]), 'Amplify(1.2x)', color='lightgreen')
btn_amp.on_clicked(lambda e: update_ui(amplify_content(im), 'Content Amplified'))

btn_rm = Button(fig.add_axes([0.73, 0.08, 0.15, 0.06]), 'Remove Object', color='lightcoral')
def handle_rm(e):
    if not np.any(current_mask): return
    ax2.set_title("Processing Object Removal..."); fig.canvas.draw_idle(); plt.pause(0.1)
    res = object_removal(im, current_mask)
    update_ui(res, "Removed & Size Restored")
    global rect_patch; 
    if rect_patch: rect_patch.remove(); rect_patch = None
    current_mask.fill(False)
btn_rm.on_clicked(handle_rm)

plt.show()