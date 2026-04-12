# import time
# import numpy as np
# import pywt
# from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from svd_engine import my_svd 

# def process_image_compression(image_file, k):
#     start_time = time.time()
#     img = Image.open(image_file).convert('RGB')
#     orig_A = np.array(img, dtype=float)
#     A = orig_A / 255.0
    
#     channels = [A[:, :, i] for i in range(3)]
#     compressed_channels = []
    
#     for channel in channels:
#         U, S, VT = my_svd(channel, k=k)
#         compressed_channels.append(U @ np.diag(S) @ VT)
    
#     compressed_img = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    
#     psnr_val = psnr(orig_A.astype(np.uint8), compressed_img)
#     ssim_val = ssim(orig_A.astype(np.uint8), compressed_img, channel_axis=2)
    
#     duration = (time.time() - start_time) * 1000
#     return compressed_img, psnr_val, ssim_val, duration

# def process_dwt_compression(image_file, threshold):
#     start_time = time.time()
    
#     img = Image.open(image_file).convert('RGB')
#     orig_size = img.size
#     orig_A = np.array(img, dtype=float)
#     A = orig_A / 255.0
    
#     compressed_channels = []
#     for i in range(3):
#         coeffs = pywt.wavedec2(A[:, :, i], 'db1', level=1)
#         coeffs_list = list(coeffs)
#         for j in range(1, len(coeffs_list)):
#             coeffs_list[j] = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs_list[j]))
#         compressed_channels.append(pywt.waverec2(coeffs_list, 'db1'))
        
#     compressed_img_raw = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
#     # 强制 resize 解决 psnr 维度报错
#     compressed_img = np.array(Image.fromarray(compressed_img_raw).resize(orig_size, Image.BILINEAR))
    
#     psnr_val = psnr(orig_A.astype(np.uint8), compressed_img)
#     ssim_val = ssim(orig_A.astype(np.uint8), compressed_img, channel_axis=2)
    
#     duration = (time.time() - start_time) * 1000
#     return compressed_img, psnr_val, ssim_val, duration

import time
import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from svd_engine import my_svd 

# 通用指标计算函数
def calc_metrics(orig_arr, comp_img):
    p = psnr(orig_arr.astype(np.uint8), comp_img)
    s = ssim(orig_arr.astype(np.uint8), comp_img, channel_axis=2)
    return p, s

def process_image_compression(image_file, k):
    start_time = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    A = orig_A / 255.0
    channels = [A[:, :, i] for i in range(3)]
    compressed_channels = []
    for channel in channels:
        U, S, VT = my_svd(channel, k=k)
        compressed_channels.append(U @ np.diag(S) @ VT)
    comp = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    p, s = calc_metrics(orig_A, comp)
    return comp, p, s, (time.time() - start_time) * 1000

def process_pca_compression(image_file, k):
    start_time = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    compressed_channels = []
    for i in range(3):
        channel = orig_A[:, :, i]
        mean = np.mean(channel, axis=0)
        U, S, VT = my_svd(channel - mean, k=k)
        compressed_channels.append((U @ np.diag(S) @ VT) + mean)
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    p, s = calc_metrics(orig_A, comp)
    return comp, p, s, (time.time() - start_time) * 1000

def process_dct_compression(image_file, k):
    start_time = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_A = np.array(img, dtype=float)
    compressed_channels = []
    for i in range(3):
        c = orig_A[:, :, i]
        dct_c = dct(dct(c.T, norm='ortho').T, norm='ortho')
        mask = np.zeros_like(dct_c)
# 增大 k 的覆盖范围，或者直接给 k 一个更大的初值范围
        mask[:k*2, :k*2] = 1
        comp = idct(idct((dct_c * mask).T, norm='ortho').T, norm='ortho')
        compressed_channels.append(comp)
    comp = np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8)
    p, s = calc_metrics(orig_A, comp)
    return comp, p, s, (time.time() - start_time) * 1000

def process_dwt_compression(image_file, threshold):
    start_time = time.time()
    img = Image.open(image_file).convert('RGB')
    orig_size, orig_A = img.size, np.array(img, dtype=float) / 255.0
    compressed_channels = []
    for i in range(3):
        coeffs = pywt.wavedec2(orig_A[:, :, i], 'db1', level=1)
        coeffs_list = list(coeffs)
        for j in range(1, len(coeffs_list)):
            coeffs_list[j] = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs_list[j]))
        compressed_channels.append(pywt.waverec2(coeffs_list, 'db1'))
    comp_raw = np.clip(np.stack(compressed_channels, axis=2) * 255, 0, 255).astype(np.uint8)
    comp = np.array(Image.fromarray(comp_raw).resize(orig_size, Image.BILINEAR))
    p, s = calc_metrics(orig_A * 255, comp)
    return comp, p, s, (time.time() - start_time) * 1000