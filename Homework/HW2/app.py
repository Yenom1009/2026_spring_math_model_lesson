# import streamlit as st
# import pandas as pd
# from PIL import Image
# from compressor import process_image_compression, process_dwt_compression

# st.set_page_config(layout="wide", page_title="图像压缩对比实验")
# st.title("📊 高性能图像压缩对比实验系统")

# # 1. 侧边栏配置
# uploaded_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
# method = st.sidebar.radio("选择压缩算法", ["SVD", "DWT (小波变换)"])

# # 动态参数显示
# if method == "SVD":
#     k = st.sidebar.slider("选择保留的 Rank (k)", 1, 200, 50)
# else:
#     threshold = st.sidebar.slider("选择 DWT 压缩阈值", 0.0, 1.0, 0.1, step=0.01)

# # 2. 图像处理逻辑
# if uploaded_file:
#     img = Image.open(uploaded_file)
    
#     with st.spinner(f"正在使用 {method} 进行计算..."):
#         if method == "SVD":
#             comp_img, psnr_val, ssim_val, duration = process_image_compression(uploaded_file, k)
#             param_label = f"Rank (k)={k}"
#         else:
#             comp_img, psnr_val, ssim_val, duration = process_dwt_compression(uploaded_file, threshold)
#             param_label = f"Threshold={threshold:.2f}"

#     # 3. 结果显示
#     st.info(f"🚀 处理完成 | 算法: {method} | 耗时: **{duration:.2f} ms**")
    
#     def get_label(p):
#         if p >= 40: return "极好", "🟢"
#         if p >= 30: return "好", "🔵"
#         if p >= 20: return "一般", "🟡"
#         return "差", "🔴"
    
#     label, icon = get_label(psnr_val)
#     st.subheader(f"当前质量评价: {icon} {label}")
    
#     # 指标展示
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("PSNR (dB)", f"{psnr_val:.2f}")
#     c2.metric("SSIM", f"{ssim_val:.4f}")
#     c3.metric("调节参数", param_label)
#     c4.metric("耗时 (ms)", f"{duration:.1f}")
    
#     st.image([img, comp_img], caption=["原始图像", f"{method} 压缩结果"], width=500)
    
#     with st.expander("点击查看评估标准表"):
#         st.table(pd.DataFrame({"PSNR (dB)": ["> 40", "30 - 40", "20 - 30", "< 20"], "质量等级": ["极好", "好", "一般", "差"]}))
# else:
#     st.warning("请在侧边栏上传图片开始实验。")

import streamlit as st
import pandas as pd
from PIL import Image
from compressor import process_image_compression, process_dwt_compression, process_pca_compression, process_dct_compression

st.set_page_config(layout="wide", page_title="图像压缩对比实验")
st.title("📊 高性能图像压缩对比实验系统")

# 1. 侧边栏配置
uploaded_file = st.sidebar.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
method = st.sidebar.radio("选择压缩算法", ["SVD", "DWT", "PCA", "DCT"])

# 统一参数逻辑
if method in ["SVD", "PCA"]:
    k = st.sidebar.slider("选择保留的 Rank (k)", 1, 200, 50)
    param_label = f"Rank={k}"
elif method == "DCT":
    k = st.sidebar.slider("DCT 低频保留大小 (k)", 1, 64, 8)
    param_label = f"Size={k}x{k}"
else:
    threshold = st.sidebar.slider("DWT 压缩阈值", 0.0, 1.0, 0.1, step=0.01)
    param_label = f"Threshold={threshold:.2f}"

# 2. 图像处理与结果显示
if uploaded_file:
    img = Image.open(uploaded_file)
    
    with st.spinner(f"正在使用 {method} 计算..."):
        if method == "SVD":
            comp_img, p, s, d = process_image_compression(uploaded_file, k)
        elif method == "PCA":
            comp_img, p, s, d = process_pca_compression(uploaded_file, k)
        elif method == "DCT":
            comp_img, p, s, d = process_dct_compression(uploaded_file, k)
        else:
            comp_img, p, s, d = process_dwt_compression(uploaded_file, threshold)

    st.info(f"🚀 处理完成 | 算法: {method} | 耗时: **{d:.2f} ms**")
    
    # 评价逻辑
    def get_label(p):
        if p >= 40: return "极好", "🟢"
        if p >= 30: return "好", "🔵"
        if p >= 20: return "一般", "🟡"
        return "差", "🔴"
    
    label, icon = get_label(p)
    st.subheader(f"当前质量评价: {icon} {label}")
    
    # 指标展示
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSNR (dB)", f"{p:.2f}")
    c2.metric("SSIM", f"{s:.4f}")
    c3.metric("调节参数", param_label)
    c4.metric("耗时 (ms)", f"{d:.1f}")
    
    st.image([img, comp_img], caption=["原始图像", f"{method} 压缩结果"], width=500)
    
    # --- 这里补回了评估标准表 ---
    with st.expander("点击查看评估标准表"):
        st.write("评估标准参考表：")
        df = pd.DataFrame({
            "PSNR (dB)": ["> 40", "30 - 40", "20 - 30", "< 20"],
            "质量等级": ["极好", "好", "一般", "差"],
            "视觉表现": ["几乎无损", "肉眼难辨", "可见模糊", "严重失真"]
        })
        st.table(df)
        
    if p > 30: st.success("结论：当前压缩参数下图像质量良好。")
    else: st.warning("结论：图像质量一般，建议增大参数以获取更多细节。")
else:
    st.warning("请在侧边栏上传图片开始实验。")