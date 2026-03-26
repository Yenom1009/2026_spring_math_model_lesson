import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RectangleSelector
from skimage import io
from skimage.transform import resize as sk_resize
from scipy.ndimage import convolve
import matplotlib.patches as patches

# 全局变量控制当前使用的能量模式
energy_mode = '1st' 

# ----------------- 核心算法：能量计算 -----------------

def get_energy_1st(im):
    """一阶能量函数 (e1)"""
    h, w = im.shape[:2]
    energy = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        channel = im[:, :, c].astype(np.float32)
        dy, dx = np.gradient(channel)
        energy += (np.abs(dx) + np.abs(dy))
    return energy

def get_energy_2nd(im):
    """二阶能量函数"""
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, -6.0, 1.0], [0.5, 1.0, 0.5]])
    h, w = im.shape[:2]
    energy = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        channel = im[:, :, c].astype(np.float32)
        laplacian = convolve(channel, kernel, mode='reflect')
        energy += laplacian ** 2
    return energy

def get_energy(im, mask=None):
    global energy_mode
    if energy_mode == '1st':
        energy = get_energy_1st(im)
    else:
        energy = get_energy_2nd(im)
        
    # 如果处于物体移除模式，给选区强制赋予极小能量
    if mask is not None:
        energy[mask] = -1e8 
    return energy

# ----------------- 核心算法：寻路与基本操作 -----------------

def get_vertical_seam(energy):
    """动态规划寻找最优接缝"""
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=int)
    for i in range(1, h):
        m_mid = M[i-1, :]
        m_l, m_r = np.roll(m_mid, 1), np.roll(m_mid, -1)
        m_l[0] = m_r[-1] = np.inf
        choices = np.stack([m_mid, m_l, m_r])
        idx = np.argmin(choices, axis=0)
        shift = np.zeros(w, dtype=int); shift[idx==1], shift[idx==2] = -1, 1
        backtrack[i, :] = (np.arange(w) + shift).astype(int)
        M[i, :] += np.min(choices, axis=0)
    seam = np.zeros(h, dtype=int)
    seam[h-1] = np.argmin(M[h-1, :])
    for i in range(h-2, -1, -1): seam[i] = backtrack[i+1, int(seam[i+1])]
    return seam.astype(int)

def remove_vertical_seam(im, seam, mask=None):
    """移除接缝"""
    h, w, c = im.shape
    new_im = np.zeros((h, w - 1, c), dtype=im.dtype)
    new_mask = np.zeros((h, w - 1), dtype=bool) if mask is not None else None
    for i in range(h):
        col = seam[i]
        new_im[i, :, :] = np.delete(im[i, :, :], col, axis=0)
        if mask is not None: new_mask[i, :] = np.delete(mask[i, :], col, axis=0)
    return (new_im, new_mask) if mask is not None else new_im

def insert_vertical_seam(im, seam, penalty):
    """插入接缝"""
    h, w, c = im.shape
    new_im = np.zeros((h, w + 1, c), dtype=im.dtype)
    new_penalty = np.zeros((h, w + 1), dtype=np.float32)
    for i in range(h):
        col = seam[i]
        p_avg = ((im[i, col, :].astype(float) + im[i, min(col+1, w-1), :].astype(float)) / 2).astype(im.dtype)
        new_im[i, :col+1, :] = im[i, :col+1, :]
        new_im[i, col+1, :] = p_avg
        new_im[i, col+2:, :] = im[i, col+1:, :]
        new_penalty[i, :col+1] = penalty[i, :col+1]
        new_penalty[i, col+2:] = penalty[i, col+1:]
        new_penalty[i, max(0, col):min(w+1, col+3)] = 1e7 # 惩罚防止在此重复插入
    return new_im, new_penalty

# ----------------- 应用：移除、放大、缩放 -----------------

def object_removal(im, mask):
    """移除物体并在完成后恢复尺寸"""
    curr_im, curr_mask = im.copy(), mask.copy()
    rows, cols = np.where(curr_mask)
    if len(rows) == 0: return curr_im
    
    # 宽>高则删横向接缝
    is_hor = (np.max(rows)-np.min(rows)) < (np.max(cols)-np.min(cols))
    if is_hor:
        curr_im, curr_mask = np.transpose(curr_im, (1, 0, 2)), np.transpose(curr_mask, (1, 0))
    
    count = 0
    # 阶段 1: 移除直到物体消失
    while np.any(curr_mask) and curr_im.shape[1] > 2:
        curr_im, curr_mask = remove_vertical_seam(curr_im, get_vertical_seam(get_energy(curr_im, curr_mask)), curr_mask)
        count += 1
    
    # 阶段 2: 恢复原始尺寸
    print(f"物体已消失，正在补回 {count} 条接缝...")
    p = np.zeros((curr_im.shape[0], curr_im.shape[1]))
    for _ in range(count):
        curr_im, p = insert_vertical_seam(curr_im, get_vertical_seam(get_energy(curr_im)+p), p)
        
    return np.transpose(curr_im, (1, 0, 2)) if is_hor else curr_im

def seam_carve_image(im, sz):
    """综合缩放功能"""
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
    """内容放大：1.2倍"""
    h, w = im.shape[:2]
    enlarged = sk_resize(im, (int(h*1.2), int(w*1.2)), preserve_range=True).astype(np.uint8)
    return seam_carve_image(enlarged, (h, w))

# ----------------- 4. GUI 界面逻辑 -----------------

try:
    im = io.imread('../figs/original.png')
except:
    im = np.random.randint(0, 255, (300, 450, 3), dtype=np.uint8)

if im.ndim == 3 and im.shape[2] == 4: im = im[:, :, :3]
current_mask = np.zeros((im.shape[0], im.shape[1]), dtype=bool)
rect_patch = None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(bottom=0.25)
ax1.imshow(im); ax1.set_title('Input Image\n(Draw Box to Remove)'); ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im)); ax2.set_title('Result'); ax2.axis('off')

# 交互组件回调
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

# 按钮 1: 能量模式切换
btn_toggle_ax = fig.add_axes([0.40, 0.08, 0.12, 0.06])
btn_toggle = Button(btn_toggle_ax, f'Energy: {energy_mode}', color='orange')

def toggle_energy(event):
    global energy_mode
    energy_mode = '2nd' if energy_mode == '1st' else '1st'
    btn_toggle.label.set_text(f'Energy: {energy_mode}')
    fig.canvas.draw_idle()

btn_toggle.on_clicked(toggle_energy)

# 按钮 2: Resize
btn_res = Button(fig.add_axes([0.54, 0.08, 0.08, 0.06]), 'Resize', color='lightblue')
def handle_resize(e):
    res = seam_carve_image(im, (int(im.shape[0]*s_row.val), int(im.shape[1]*s_col.val)))
    himg.set_data(res); himg.set_extent([0, res.shape[1], res.shape[0], 0])
    ax2.set_title(f'Resized ({energy_mode})'); fig.canvas.draw_idle()
btn_res.on_clicked(handle_resize)

# 按钮 3: Amplify
btn_amp = Button(fig.add_axes([0.64, 0.08, 0.12, 0.06]), 'Amplify(1.2x)', color='lightgreen')
def handle_amp(e):
    res = amplify_content(im)
    himg.set_data(res); himg.set_extent([0, res.shape[1], res.shape[0], 0])
    ax2.set_title(f'Amplify ({energy_mode})'); fig.canvas.draw_idle()
btn_amp.on_clicked(handle_amp)

# 按钮 4: Remove Object
btn_rm = Button(fig.add_axes([0.78, 0.08, 0.14, 0.06]), 'Remove Object', color='lightcoral')
def handle_rm(e):
    if not np.any(current_mask): return
    ax2.set_title("Processing Object Removal..."); fig.canvas.draw_idle(); plt.pause(0.1)
    res = object_removal(im, current_mask)
    himg.set_data(res); himg.set_extent([0, res.shape[1], res.shape[0], 0])
    ax2.set_title(f'Removed & Restored ({energy_mode})')
    global rect_patch
    if rect_patch: rect_patch.remove(); rect_patch = None
    current_mask.fill(False); fig.canvas.draw_idle()
btn_rm.on_clicked(handle_rm)

plt.show()