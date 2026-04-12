import numpy as np
import matplotlib.pyplot as plt
import time
import io
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  
from svd_engine import my_svd 

# --- 核心：与 compressor1.py 保持一致的真实体积计算函数 ---
def get_real_storage_size(u_list, s_list, vt_list, k):
    """模拟真实二进制存储体积"""
    buf = io.BytesIO()
    storage_data = {}
    for i in range(3):
        # 截取前 k 个分量并转为 float32 (工业级 4 字节标准)
        storage_data[f'u_{i}'] = u_list[i][:, :k].astype(np.float32)
        storage_data[f's_{i}'] = s_list[i][:k].astype(np.float32)
        storage_data[f'vt_{i}'] = vt_list[i][:k, :].astype(np.float32)
    
    np.savez_compressed(buf, **storage_data)
    return len(buf.getvalue())

def run_analysis(image_path, k_range):
    # 1. 加载并缩放图片 (为了分析效率，统一缩放到 300px 左右)
    img = Image.open(image_path).convert('RGB')
    if max(img.size) > 300:
        img = img.resize((300, 300))
    
    orig_A = np.array(img, dtype=float)
    orig_bytes = orig_A.nbytes # 原始 uint8 字节数
    H, W, C = orig_A.shape
    A_norm = orig_A / 255.0
    
    # 2. 预先计算全量 SVD (全量分解只做一次)
    print(f"正在进行初始全量 SVD 分解 (图片尺寸: {H}x{W})...")
    u_all, s_all, vt_all = [], [], []
    for i in range(3):
        U, S, VT = my_svd(A_norm[:, :, i])
        u_all.append(U)
        s_all.append(S)
        vt_all.append(VT)
    
    psnr_list, ssim_list, cr_list, time_list = [], [], [], []
    
    # 3. 扫描 k 值
    print("\n开始扫描 k 值并根据真实存储逻辑计算指标:")
    for k in tqdm(k_range, desc="📊 绘图进度", unit="k"):
        start_t = time.time()
        
        # --- 重建图像用于计算质量指标 ---
        comp_channels = []
        for i in range(3):
            Ak = u_all[i][:, :k] @ np.diag(s_all[i][:k]) @ vt_all[i][:k, :]
            comp_channels.append(Ak)
        compressed_img = np.clip(np.stack(comp_channels, axis=2) * 255, 0, 255).astype(np.uint8)
        
        # 计算质量指标
        p = psnr(orig_A.astype(np.uint8), compressed_img)
        s = ssim(orig_A.astype(np.uint8), compressed_img, channel_axis=2)
        
        # --- 计算真实压缩比 (同步最新逻辑) ---
        real_comp_size = get_real_storage_size(u_all, s_all, vt_all, k)
        cr = orig_bytes / real_comp_size
        
        duration = (time.time() - start_t) * 1000 
        
        psnr_list.append(p)
        ssim_list.append(s)
        cr_list.append(cr)
        time_list.append(duration)
        
    return psnr_list, ssim_list, cr_list, time_list

# --- 设置运行参数 ---
image_to_test = "peppers.jpg" # 请确保图片在当前目录
ks = np.arange(1, 201, 5) # 扫描步长

p_data, s_data, c_data, t_data = run_analysis(image_to_test, ks)

# --- 绘图逻辑 ---
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

titles = ['PSNR (质量) 随 k 值变化', 'SSIM (感知) 随 k 值变化', '真实压缩比 (CR) 随 k 变化', '重构耗时 (ms) 随 k 变化']
data_sets = [p_data, s_data, c_data, t_data]
y_labels = ['PSNR (dB)', 'SSIM 指数', '压缩比 (X : 1)', '耗时 (ms)']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'] # 更美观的配色

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(ks, data_sets[i], color=colors[i], marker='o', markersize=3, linewidth=1.5)
    plt.title(titles[i], fontsize=12, fontweight='bold')
    plt.xlabel('截断秩 k')
    plt.ylabel(y_labels[i])
    plt.grid(True, linestyle='--', alpha=0.6)

plt.suptitle(f"SVD 图像压缩性能深度分析 ({image_to_test})", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('svd_final_analysis.png', dpi=300)
print("\n✅ 分析图表已保存为 'svd_final_analysis.png'，可直接插入实验报告。")
plt.show()