import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from compressor3 import *

st.set_page_config(layout="wide", page_title="图像压缩对比系统")

def get_params_from_target_cr(img_shape, target_cr):
    H, W, _ = img_shape
    total_pixels_per_channel = H * W
    # 目标每通道允许的浮点数个数 = (H*W*1字节) / (target_cr * 4字节)
    target_values = total_pixels_per_channel / (target_cr * 4)
    
    # 1. SVD/PCA: k*(H+W) = target_values -> k = target_values / (H+W)
    k_svd = int(target_values / (H + W))
    k_svd = max(1, min(k_svd, 200))
    
    # 2. DCT: (H*ratio)*(W*ratio) = target_values -> ratio = sqrt(target_values / (H*W))
    ratio = np.sqrt(target_values / (H * W))
    k_dct = int(ratio * 64)
    k_dct = max(1, min(k_dct, 64))
    
    # 3. DWT: 阈值很难反推，给一个经验线性值
    threshold_dwt = (target_cr / 100.0) * 0.4
    threshold_dwt = max(0.001, min(threshold_dwt, 0.8))
    
    return k_svd, k_dct, threshold_dwt

def get_label(p):
    if p >= 40: return "极好 (几乎无损)", "🟢"
    elif p >= 30: return "好 (肉眼难辨)", "🔵"
    elif p >= 20: return "一般 (可见模糊)", "🟡"
    return "差 (严重失真)", "🔴"

st.title("📊 高性能图像压缩公平对比系统")
st.markdown("计算标准：压缩比 = 原始像素字节(uint8) / 重构必需参数字节(float32)")

mode = st.sidebar.selectbox("功能模式选择", ["单算法详细调节", "多算法目标压缩比对比"])
uploaded_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    if mode == "单算法详细调节":
        st.sidebar.markdown("---")
        method = st.sidebar.radio("选择算法", ["SVD", "DWT", "PCA", "DCT"])
        if method in ["SVD", "PCA"]:
            k = st.sidebar.slider("选择 Rank (k)", 1, 200, 50)
        elif method == "DCT":
            k = st.sidebar.slider("DCT 频率保留大小 (1-64)", 1, 64, 32)
        else:
            threshold = st.sidebar.slider("DWT 阈值", 0.001, 0.8, 0.05)

        with st.spinner("计算中..."):
            if method == "SVD": res = process_image_compression(uploaded_file, k)
            elif method == "PCA": res = process_pca_compression(uploaded_file, k)
            elif method == "DCT": res = process_dct_compression(uploaded_file, k)
            else: res = process_dwt_compression(uploaded_file, threshold)
        
        comp_img, p, s, cr, d = res
        label, icon = get_label(p)
        st.subheader(f"当前评价: {icon} {label}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PSNR (dB)", f"{p:.2f}")
        c2.metric("SSIM", f"{s:.4f}")
        c3.metric("公平压缩比", f"{cr:.2f}:1")
        c4.metric("耗时 (ms)", f"{d:.1f}")
        st.image([img, comp_img], caption=["原图", f"{method} 结果"], width=500)

    else: # 对比模式
        target_cr = st.sidebar.slider("设定预期压缩比 (Target CR)", 1.5, 50.0, 10.0, step=0.5)
        if st.sidebar.button("开始全算法性能对比"):
            k_svd, k_dct, th_dwt = get_params_from_target_cr(img_array.shape, target_cr)
            
            with st.spinner("算法同步计算中..."):
                results = [
                    ("SVD", process_image_compression(uploaded_file, k_svd)),
                    ("PCA", process_pca_compression(uploaded_file, k_svd)),
                    ("DCT", process_dct_compression(uploaded_file, k_dct)),
                    ("DWT", process_dwt_compression(uploaded_file, th_dwt))
                ]
            
            st.subheader(f"在目标 CR ≈ {target_cr}:1 下的表现")
            idx = 0
            for row in range(2):
                cols = st.columns(2)
                for col in range(2):
                    name, (c_img, p, s, cr, d) = results[idx]
                    with cols[col]:
                        st.image(c_img, caption=f"{name} (实际 CR: {cr:.2f}:1)", use_container_width=True)
                        st.caption(f"PSNR: {p:.2f} | SSIM: {s:.4f}")
                    idx += 1
            
            st.table(pd.DataFrame({
                "算法": [r[0] for r in results],
                "PSNR": [f"{r[1][1]:.2f}" for r in results],
                "SSIM": [f"{r[1][2]:.4f}" for r in results],
                "实际CR": [f"{r[1][3]:.2f}:1" for r in results]
            }))