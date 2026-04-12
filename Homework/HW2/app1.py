import streamlit as st
import pandas as pd
from PIL import Image
from compressor1 import * # 请确保这里的导入文件名和你实际文件名一致

st.set_page_config(layout="wide", page_title="图像压缩实验系统")
st.title("📊 高性能图像压缩对比实验系统")

# 1. 侧边栏配置
uploaded_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
method = st.sidebar.radio("选择压缩算法", ["SVD", "DWT", "PCA", "DCT"])

if method in ["SVD", "PCA"]:
    k = st.sidebar.slider("选择 Rank (k)", 1, 200, 50)
    param_label = f"Rank={k}"
elif method == "DCT":
    k = st.sidebar.slider("DCT 频率块大小", 1, 64, 8)
    param_label = f"Size={k}x{k}"
else:
    threshold = st.sidebar.slider("DWT 阈值", 0.0, 1.0, 0.1, step=0.01)
    param_label = f"Threshold={threshold:.2f}"

# 2. 图像处理与指标计算
if uploaded_file:
    img = Image.open(uploaded_file)
    with st.spinner(f"正在进行 {method} 计算..."):
        if method == "SVD":
            res = process_image_compression(uploaded_file, k)
        elif method == "PCA":
            res = process_pca_compression(uploaded_file, k)
        elif method == "DCT":
            res = process_dct_compression(uploaded_file, k)
        else:
            res = process_dwt_compression(uploaded_file, threshold)
    
    comp_img, p, s, cr, d = res
    st.info(f"🚀 处理成功 | 算法: {method} | 耗时: **{d:.2f} ms**")
    
    # --- 核心评价逻辑 (被删掉的部分已补回) ---
    def get_label(p):
        if p >= 40: return "极好 (几乎无损)", "🟢"
        if p >= 30: return "好 (肉眼难辨)", "🔵"
        if p >= 20: return "一般 (可见模糊)", "🟡"
        return "差 (严重失真)", "🔴"
    
    label, icon = get_label(p)
    st.subheader(f"当前质量评价: {icon} {label}")
    
    # 3. 指标 metric 显示
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSNR (dB)", f"{p:.2f}")
    c2.metric("SSIM", f"{s:.4f}")
    c3.metric("压缩比 (CR)", f"{cr:.1f}:1")
    c4.metric("计算耗时 (ms)", f"{d:.1f}")
    
    st.image([img, comp_img], caption=["原始图像", f"{method} 压缩结果 (参数: {param_label})"], width=500)
    
    # 4. 辅助建议与评价标准表
    if p > 30: st.success("结论：当前压缩参数下图像质量保持良好。")
    else: st.warning("结论：图像质量一般，建议增大参数以获取更多细节。")
    
    with st.expander("点击查看评估标准表"):
        st.table(pd.DataFrame({
            "PSNR (dB)": ["> 40", "30 - 40", "20 - 30", "< 20"],
            "质量等级": ["极好", "好", "一般", "差"]
        }))
else:
    st.warning("请在侧边栏上传图片开始实验。")