import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from compressor2 import *

st.set_page_config(layout="wide", page_title="图像压缩实验系统")

# --- 辅助功能：根据目标压缩比推算各算法参数 ---
def get_params_from_target_cr(img_shape, target_cr):
    H, W, _ = img_shape
    total_pixels = H * W
    
    # 1. SVD/PCA 参数 k 计算 (基于理论公式反推作为起始点)
    # 理论 CR = (H*W) / (k*(H+W+1)) -> k = (H*W) / (CR * (H+W+1))
    k_svd = int(total_pixels / (target_cr * (H + W + 1)))
    k_svd = max(1, min(k_svd, 200))
    
    # 2. DCT 参数 k 计算
    # 理论 CR = (H*W) / k^2 -> k = sqrt(H*W / CR)
    # 注意：我们的DCT k对应的是比例，这里做一个映射转换
    k_dct = int(np.sqrt(total_pixels / target_cr) / (max(H,W)/64))
    k_dct = max(1, min(k_dct, 64))
    
    # 3. DWT 参数 threshold 计算 (经验线性映射)
    # 压缩比越高，阈值越大
    threshold_dwt = target_cr * 0.005 
    threshold_dwt = max(0.001, min(threshold_dwt, 0.5))
    
    return k_svd, k_dct, threshold_dwt

# --- 核心评价等级函数 ---
def get_label(p):
    if p >= 40: return "极好 (几乎无损)", "🟢"
    elif p >= 30: return "好 (肉眼难辨)", "🔵"
    elif p >= 20: return "一般 (可见模糊)", "🟡"
    return "差 (严重失真)", "🔴"

# --- UI 界面渲染 ---
st.title("📊 高性能图像压缩对比实验系统")

# 侧边栏：功能模式选择
mode = st.sidebar.selectbox("功能模式选择", ["单算法详细调节", "多算法目标压缩比对比"])
uploaded_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    if mode == "单算法详细调节":
        st.sidebar.markdown("---")
        method = st.sidebar.radio("选择算法", ["SVD", "DWT", "PCA", "DCT"])
        
        # 参数控件
        if method in ["SVD", "PCA"]:
            k = st.sidebar.slider("选择 Rank (k)", 1, 200, 50)
            param_label = f"Rank={k}"
        elif method == "DCT":
            k = st.sidebar.slider("DCT 频率块大小", 1, 64, 8)
            param_label = f"Size={k}x{k}"
        else:
            threshold = st.sidebar.slider("DWT 阈值", 0.0, 1.0, 0.1, step=0.01)
            param_label = f"Threshold={threshold:.2f}"

        # 执行计算
        with st.spinner(f"正在使用 {method} 计算..."):
            if method == "SVD": res = process_image_compression(uploaded_file, k)
            elif method == "PCA": res = process_pca_compression(uploaded_file, k)
            elif method == "DCT": res = process_dct_compression(uploaded_file, k)
            else: res = process_dwt_compression(uploaded_file, threshold)
        
        comp_img, p, s, cr, d = res
        
        # 显示结果
        st.info(f"🚀 处理成功 | 算法: {method} | 耗时: **{d:.2f} ms**")
        label, icon = get_label(p)
        st.subheader(f"当前质量评价: {icon} {label}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PSNR (dB)", f"{p:.2f}")
        c2.metric("SSIM", f"{s:.4f}")
        c3.metric("实际压缩比 (CR)", f"{cr:.1f}:1")
        c4.metric("计算耗时 (ms)", f"{d:.1f}")
        
        st.image([img, comp_img], caption=["原始图像", f"{method} 压缩结果 (参数: {param_label})"], width=500)
        
        if p > 30: st.success("结论：当前参数下质量良好。")
        else: st.warning("结论：失真较明显，建议调整参数。")

    else: # 多算法目标压缩比对比
        st.sidebar.markdown("---")
        target_cr = st.sidebar.slider("预期压缩比 (Target CR)", 2.0, 100.0, 15.0, step=1.0)
        
        if st.sidebar.button("开始全算法对比"):
            k_svd, k_dct, th_dwt = get_params_from_target_cr(img_array.shape, target_cr)
            
            results = []
            with st.spinner("正在并行处理四种算法..."):
                results.append(("SVD", process_image_compression(uploaded_file, k_svd)))
                results.append(("PCA", process_pca_compression(uploaded_file, k_svd)))
                results.append(("DCT", process_dct_compression(uploaded_file, k_dct)))
                results.append(("DWT", process_dwt_compression(uploaded_file, th_dwt)))
            
            # 展示四宫格结果
            st.subheader(f"目标压缩比 ≈ {target_cr}:1 下的各算法表现")
            idx = 0
            for row in range(2):
                cols = st.columns(2)
                for col in range(2):
                    name, (c_img, p, s, cr, d) = results[idx]
                    with cols[col]:
                        st.image(c_img, caption=f"{name} (Actual CR: {cr:.1f}:1)", use_container_width=True)
                        st.write(f"**PSNR:** {p:.2f} | **SSIM:** {s:.4f}")
                    idx += 1
            
            # 对比汇总表
            st.markdown("### 性能汇总对比")
            summary_data = {
                "算法": [r[0] for r in results],
                "PSNR (dB)": [r[1][1] for r in results],
                "SSIM": [r[1][2] for r in results],
                "实际压缩比": [f"{r[1][3]:.1f}:1" for r in results],
                "耗时 (ms)": [f"{r[1][4]:.1f}" for r in results]
            }
            st.table(pd.DataFrame(summary_data))

    # 公共组件：评估标准表
    with st.expander("点击查看评估标准参考"):
        st.table(pd.DataFrame({
            "PSNR (dB)": ["> 40", "30 - 40", "20 - 30", "< 20"],
            "质量等级": ["极好", "好", "一般", "差"],
            "视觉表现": ["几乎无损", "肉眼难辨", "可见模糊", "严重失真"]
        }))
else:
    st.warning("请在侧边栏上传图片开始实验。")