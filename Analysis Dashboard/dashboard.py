import sys
from pathlib import Path

# Add project root and current dir to sys.path for seamless imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from modules.eda import dataset_overview, handle_missing_values, correlation_matrix, top_correlated_features, kpi_and_visualization, remove_outliers_isolation_forest
from modules.eda import display_textual_outlier_comparison, visualize_top_features_box_plots, log_transform, normalize_data, perform_normality_tests
from modules.eda import plot_histograms_before_after, plot_qq_before_after_with_plotly, apply_transformations, perform_t_test
from modules.eda import perform_pca, plot_pca_2d, pca_analysis, train_and_compare_classification_models, show_textual_report, show_visualizations

# Database connection path
DB_PATH = CURRENT_DIR / 'radiomics_data.db'

@st.cache_data(ttl=7200)
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT * FROM radiomic_features"
        return pd.read_sql(query, conn)

# Page configuration
st.set_page_config(page_title="COVID-19 Radiomics Dashboard", layout="wide")

# Custom CSS for blue-shaded dark-themed styling
blue_theme_css = """
    <style>
        .main-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #FFFFFF;
            text-align: center;
            padding: 10px;
        }
        .kpi-card {
            background-color: #2E3B55;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #2E3B55;
            color: #FFFFFF;
        }
        .dark-theme {
            background-color: #2E3B55;
            color: #FFFFFF;
        }
        .output-container {
            margin-top: 30px;
        }
        .main-container {
            background-color: #243B53;
        }
    </style>
"""

# Apply blue shade theme CSS
st.markdown(blue_theme_css, unsafe_allow_html=True)

st.sidebar.title("COVID-19 Radiomics Dashboard")
page_selection = st.sidebar.radio("Navigate to:", ["Textual Analysis", "Visual Comparisons", "New Diagnosis (Chẩn đoán ảnh mới)"])
# Normalization Methods in the sidebar
normalization_method = st.sidebar.radio(
    "Select Normalization Method",
    ('Min-Max', 'Z-score', 'Max-Abs')
)


 # Load and filter data based on selection
data_df = load_data()
if "Target" in data_df.columns:
    data_df["Target"] = data_df["Target"].astype(str)
else:
    st.error("The 'Target' column is missing from the dataset.")
    st.stop()

# Convert all object-type columns to string and numeric columns to float
for col in data_df.columns:
    if data_df[col].dtype == 'O':
        data_df[col] = data_df[col].astype(str).fillna('')
    else:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
feature_classes = {
    'firstorder': ['Entropy', 'Energy', 'Uniformity', 'MeanAbsoluteDeviation', 'Skewness', 'Kurtosis'],
    'glcm': ['Contrast', 'Idm', 'Correlation', 'ClusterProminence', 'ClusterShade'],
    'gldm': ['GrayLevelNonUniformity', 'DependenceNonUniformity', 'DependenceVariance', 'LowGrayLevelEmphasis', 'HighGrayLevelEmphasis'],
    'glszm': ['ZoneEntropy', 'LargeAreaEmphasis', 'SmallAreaEmphasis', 'GrayLevelVariance', 'SizeZoneNonUniformity'],
    'glrlm': ['RunLengthNonUniformity', 'ShortRunEmphasis', 'LongRunEmphasis']
}

# Generate a list of selected columns based on the feature classes
selected_columns = []
for key, features in feature_classes.items():
    selected_columns.extend([f'original_{key}_{feature}' for feature in features])

selected_columns.append('Target')
        
if page_selection == "Textual Analysis":
    st.title("Textual Analysis")
    st.write("This page includes dataset summaries, descriptive statistics, and hypothesis testing for selected features.")
    
    # Filter option only for textual analysis page
    analysis_type = st.sidebar.selectbox("Select Data Type", options=["Both", "COVID", "Normal"])

    # Load and filter data based on selection
    data_df = load_data()
    if "Target" in data_df.columns:
        data_df["Target"] = data_df["Target"].astype(str)
    else:
        st.error("The 'Target' column is missing from the dataset.")
        st.stop()

    if analysis_type == "COVID":
        data_df = data_df[data_df["Target"] == '1']
    elif analysis_type == "Normal":
        data_df = data_df[data_df["Target"] == '0']

    # Convert all object-type columns to string and numeric columns to float
    for col in data_df.columns:
        if data_df[col].dtype == 'O':
            data_df[col] = data_df[col].astype(str).fillna('')
        else:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # Show the dataset overview based on selected filter
    dataset_overview(data_df, analysis_type)

    # Display the filtered data preview
    st.markdown(f'<div class="output-container"><h3>Filtered Data for {analysis_type} Cases</h3></div>', unsafe_allow_html=True)
    st.write(data_df.head())

    # Handle empty DataFrame case
    if data_df.empty:
        st.error(f"No data available for {analysis_type} filter. Please verify the dataset or the filter selection.")
        st.stop()

    # Handle missing values and EDA
    data_df = handle_missing_values(data_df)
    # KPI Section
    st.markdown('<div class="main-header">Descriptive Statistics for Selected Feature</div>', unsafe_allow_html=True)

    # Numerical column selection
    numerical_columns = data_df.select_dtypes(include=[np.number]).columns
    if not numerical_columns.empty:
        selected_feature = st.selectbox("Select a Feature for Analysis", options=numerical_columns)

        # Calculate statistics for the selected feature
        mean_value = data_df[selected_feature].mean()
        median_value = data_df[selected_feature].median()
        mode_value = data_df[selected_feature].mode().iloc[0] if not data_df[selected_feature].mode().empty else "No Mode"
        std_dev_value = data_df[selected_feature].std()
        iqr_value = data_df[selected_feature].quantile(0.75) - data_df[selected_feature].quantile(0.25)

        # Display KPI Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="kpi-card">Mean<br>{mean_value:.2f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card">Median<br>{median_value:.2f}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card">Mode<br>{mode_value}</div>', unsafe_allow_html=True)

        # Variability Analysis Section
        st.markdown('<div class="main-header">Variability Analysis for Selected Feature</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="kpi-card">Standard Deviation<br>{std_dev_value:.2f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card">Interquartile Range (IQR)<br>{iqr_value:.2f}</div>', unsafe_allow_html=True)
    else:
        st.error("No numerical columns available for analysis.")

    # Compute correlation matrix
    corr_matrix = correlation_matrix(data_df, selected_columns)

    # Display correlation matrix in textual format
    st.write("### Correlation Matrix ")
    st.write(corr_matrix)

    # Identify top 5 correlated features with the target
    target_column = 'Target'
    top_features, top_correlations = top_correlated_features(corr_matrix, target_column)
    if top_correlations is not None:
        st.write("### Top 5 Features Most Correlated with Target ")
        st.write(top_correlations[1:])  # Exclude the target column itself

    filtered_data = remove_outliers_isolation_forest(data_df)
        
    display_textual_outlier_comparison(data_df, filtered_data, top_features)

    # Normalize data with Min-Max or Z-score
    normalized_df = normalize_data(data_df, selected_columns, normalization_method)

    transformed_df = apply_transformations(data_df, selected_columns)
    
    normality_results = perform_normality_tests(data_df, transformed_df, selected_columns)
    
    perform_t_test(data_df,top_features)
    
    pca_analysis(data_df)
    
    model_result = train_and_compare_classification_models(data_df, top_features)
    
    show_textual_report(model_result)
    
    

elif page_selection == "Visual Comparisons":
    st.title("Visual Comparisons")
    st.write("This page includes before-and-after visualizations for outlier removal and normalization.")
    col1, col2 = st.columns(2)
    
    with col1:
        corr_matrix = correlation_matrix(data_df, selected_columns)
        # Display heatmap for correlation matrix using plotly
        st.write("### Correlation Matrix Heatmap")
        fig_heatmap = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_heatmap.update_layout(title="Correlation Matrix for Selected Features")
        st.plotly_chart(fig_heatmap)
        
    with col2:
        top_features, top_correlations = top_correlated_features(corr_matrix)
        # Display bar chart for top 5 correlated features
        if top_correlations is not None:
            st.write("### Top 5 Features Most Correlated with Target ")
            top_features_values = top_correlations[1:]  # Exclude the target column itself
            fig_bar = px.bar(top_features_values, x=top_features_values.index, y=top_features_values.values,
                            labels={'y': 'Correlation', 'index': 'Features'})
            fig_bar.update_layout(title="Top 5 Correlated Features with Target")
            st.plotly_chart(fig_bar)
        
    kpi_and_visualization(data_df, selected_columns)
    
    filtered_data = remove_outliers_isolation_forest(data_df)

    transformed_data = log_transform(filtered_data)
    
    visualize_top_features_box_plots(transformed_data, top_features)
    
    # Normalize data with Min-Max or Z-score
    normalized_df = normalize_data(data_df, selected_columns, normalization_method)
    
    # Plot QQ plots before and after normalization
    plot_qq_before_after_with_plotly(data_df, normalized_df, top_features)
    
    # Plot Histograms Before and After Normalization
    plot_histograms_before_after(data_df, normalized_df, top_features)
    
    pca, pca_df = perform_pca(data_df,n_components=2)
    
    plot_pca_2d(pca_df, data_df)
    
    model_result = train_and_compare_classification_models(data_df, top_features)
    
    show_visualizations(model_result, top_features)

elif page_selection == "New Diagnosis (Chẩn đoán ảnh mới)":
    st.title("🩺 Hệ thống Hỗ trợ Chẩn đoán Lâm sàng (Clinical CAD Diagnosis)")
    st.write("Hỗ trợ phân tích ảnh đơn lẻ hoặc toàn bộ chuỗi cắt lớp vi tính CT Scan Lồng ngực / X-quang (DICOM, PNG, JPG), tự động phân vùng nhu mô phổi (U-Net), trích xuất 24 đặc trưng Radiomics và dự đoán tổn thương dạng COVID-19 vs. Normal.")

    import io
    import re
    from modules.inference import diagnose_image, create_cine_gif, load_dicom_image, is_dicom_input
    from PIL import Image

    def natural_sort_key(s):
        """Sort slice filenames in natural numerical order (e.g. 52, 53, ..., 62)."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    uploaded_files = st.file_uploader(
        "Tải lên ảnh chụp CT Lồng ngực / X-quang (Hỗ trợ tải lên nhiều ảnh hoặc toàn bộ series DICOM .dcm, PNG, JPG):",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Natural sort so slices are ordered correctly
        uploaded_files = sorted(uploaded_files, key=lambda f: natural_sort_key(f.name))
        total_uploaded = len(uploaded_files)

        # Cache management in session_state
        files_sig = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("diagnosis_files_sig") != files_sig:
            st.session_state["diagnosis_results"] = None
            st.session_state["diagnosis_files_sig"] = files_sig

        col_cfg1, col_cfg2 = st.columns([1, 1])
        with col_cfg1:
            gif_duration = st.slider(
                "⏱️ Tốc độ chuyển lát cắt trong ảnh GIF (ms/frame):",
                min_value=100, max_value=800, value=300, step=50,
                help="Thời lượng hiển thị mỗi lát cắt CT trong ảnh động Cine-Loop GIF (300ms tương đương ~3.3 fps)."
            )
        with col_cfg2:
            gif_mode = st.radio(
                "🎨 Kiểu hiển thị ảnh động Cine-Loop GIF:",
                ["Song song (Ảnh CT Gốc | Lớp phủ Phân vùng Cyan)", "Chỉ Lớp phủ Phân vùng (Overlay Mask)"],
                horizontal=True
            )
        is_side_by_side = ("Song song" in gif_mode)

        # Preview of first slice
        first_file = uploaded_files[0]
        first_bytes = first_file.getvalue()
        first_is_dcm = first_file.name.lower().endswith(('.dcm', '.dicom')) or is_dicom_input(io.BytesIO(first_bytes))
        
        with st.expander(f"📁 Danh sách {total_uploaded} lát cắt đã tải lên (Xem trước lát đầu: `{first_file.name}`)", expanded=(st.session_state.get("diagnosis_results") is None)):
            prev_col1, prev_col2 = st.columns([1, 2])
            with prev_col1:
                try:
                    if first_is_dcm:
                        preview_arr, dcm_meta = load_dicom_image(io.BytesIO(first_bytes))
                        st.image(Image.fromarray(preview_arr), caption=f"Lát #{1}: {first_file.name}", width=220)
                    else:
                        st.image(Image.open(io.BytesIO(first_bytes)), caption=f"Lát #{1}: {first_file.name}", width=220)
                except Exception as prev_err:
                    st.warning(f"Không thể tạo ảnh xem trước: {prev_err}")
            with prev_col2:
                st.markdown(f"**Tổng số file đã chọn:** `{total_uploaded}` lát cắt")
                st.markdown(f"**Series từ:** `{uploaded_files[0].name}` ➡️ `{uploaded_files[-1].name}`")
                if first_is_dcm:
                    st.success("✅ Định dạng: DICOM y tế tiêu chuẩn (Tự động áp dụng cửa sổ nhu mô phổi WL=-600, WW=1500 HU)")
                else:
                    st.info("ℹ️ Định dạng: Ảnh 2D tiêu chuẩn (PNG/JPG)")
                st.write("Nhấn nút bên dưới để khởi chạy toàn bộ quy trình: Phân vùng nhu mô phổi (U-Net) ➡️ Trích xuất 24 đặc trưng Radiomics ➡️ Dự đoán phân loại bằng Random Forest.")

        run_diag = st.button("🚀 Thực hiện Chẩn đoán Lâm sàng Toàn diện", type="primary", use_container_width=True)

        if run_diag:
            from datetime import datetime
            import shutil
            import subprocess

            # Setup output directories
            root_dir = Path(__file__).resolve().parent.parent
            output_base = root_dir / "output"
            output_base.mkdir(parents=True, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = output_base / f"diagnosis_session_{timestamp_str}"
            uploads_dir = session_dir / "uploads"
            masks_dir = session_dir / "masks"
            overlays_dir = session_dir / "overlays"

            uploads_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            overlays_dir.mkdir(parents=True, exist_ok=True)

            results = []
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for idx, uf in enumerate(uploaded_files):
                status_text.info(f"⏳ Đang phân tích và lưu trữ lát cắt {idx + 1}/{total_uploaded}: `{uf.name}`...")
                progress_bar.progress((idx + 1) / total_uploaded)

                f_bytes = uf.getvalue()

                # 1. Save uploaded original file to disk
                saved_upload_path = uploads_dir / uf.name
                saved_upload_path.write_bytes(f_bytes)

                # 2. Run inference
                stream = io.BytesIO(f_bytes)
                stream.name = uf.name
                res = diagnose_image(stream, filename=uf.name)
                res["filename"] = uf.name
                res["slice_index"] = idx + 1
                res["saved_upload_path"] = str(saved_upload_path)

                # 3. Save predicted U-Net mask
                stem = Path(uf.name).stem
                mask_path = masks_dir / f"{stem}_mask.png"
                Image.fromarray(res["mask"]).save(mask_path)
                res["saved_mask_path"] = str(mask_path)

                # 4. Save cyan lesion overlay
                overlay_path = overlays_dir / f"{stem}_overlay.png"
                Image.fromarray(res["overlay"]).save(overlay_path)
                res["saved_overlay_path"] = str(overlay_path)

                results.append(res)

            # 5. Generate and save Cine-Loop GIF
            gif_bytes = create_cine_gif(results, duration_ms=gif_duration, side_by_side=is_side_by_side)
            gif_file_path = session_dir / f"patient_cineloop_{total_uploaded}_slices.gif"
            gif_file_path.write_bytes(gif_bytes)

            # 6. Save batch diagnosis summary CSV
            summary_rows = []
            for r in results:
                summary_rows.append({
                    "Lát cắt": f"Slice #{r['slice_index']}",
                    "Tên file": r["filename"],
                    "Chẩn đoán": r["prediction"],
                    "Độ tin cậy": f"{r['confidence']:.2f}%",
                    "Xác suất Bình thường": f"{r['probability_normal']*100:.2f}%",
                    "Xác suất COVID-19": f"{r['probability_covid']*100:.2f}%",
                    "File ảnh gốc": str(Path(r.get("saved_upload_path", "")).name),
                    "File Mask": str(Path(r.get("saved_mask_path", "")).name),
                    "File Overlay": str(Path(r.get("saved_overlay_path", "")).name)
                })
            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = session_dir / "batch_diagnosis_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

            # 7. Save clinical report text
            t_slices = len(results)
            n_cnt = sum(1 for r in results if r["prediction"] == "Normal")
            c_cnt = sum(1 for r in results if r["prediction"] == "COVID-19")
            avg_n_prob = float(np.mean([r["probability_normal"] for r in results])) * 100
            avg_c_prob = float(np.mean([r["probability_covid"] for r in results])) * 100
            avg_c_conf = float(np.mean([r["confidence"] for r in results]))

            if c_cnt == 0:
                concl_text = f"Kết luận lâm sàng: Cả {t_slices} lát cắt của bệnh nhân đều được chẩn đoán là NORMAL (Âm tính với tổn thương đông đặc/kính mờ dạng COVID-19), với xác suất trung bình {avg_n_prob:.1f}%."
            else:
                concl_text = f"Kết luận lâm sàng: Phát hiện {c_cnt}/{t_slices} lát cắt có dấu hiệu tổn thương nghi ngờ COVID-19 (Đông đặc/kính mờ), với xác suất trung bình {avg_c_prob:.1f}%. Cần bác sĩ hội chẩn lâm sàng chi tiết!"

            report_path = session_dir / "clinical_report.txt"
            report_content = f"""=======================================================
HỆ THỐNG CAD PHỔI - BÁO CÁO KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG
=======================================================
Thời gian chẩn đoán : {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
Thư mục lưu trữ     : {session_dir.resolve()}
Tổng số lát cắt CT  : {t_slices}

{concl_text}

THỐNG KÊ CHI TIẾT:
- Số lát Bình thường (Normal)    : {n_cnt}/{t_slices} ({n_cnt/t_slices*100:.1f}%)
- Số lát nghi ngờ COVID-19       : {c_cnt}/{t_slices} ({c_cnt/t_slices*100:.1f}%)
- Xác suất Bình thường trung bình: {avg_n_prob:.1f}%
- Xác suất COVID-19 trung bình   : {avg_c_prob:.1f}%
- Độ tin cậy trung bình          : {avg_c_conf:.1f}%

DANH MỤC TÀI NGUYÊN ĐÃ LƯU TRỮ:
- 📁 Ảnh gốc đã tải lên        : uploads/ ({t_slices} files)
- 🎭 Mặt nạ phân vùng U-Net     : masks/ ({t_slices} files)
- 🎨 Lớp phủ tổn thương Cyan   : overlays/ ({t_slices} files)
- 🎞️ Ảnh động Cine-Loop GIF     : {gif_file_path.name}
- 📊 Bảng tổng hợp CSV         : batch_diagnosis_summary.csv
=======================================================
"""
            report_path.write_text(report_content, encoding="utf-8")

            # Update latest copy
            latest_dir = output_base / "latest"
            if latest_dir.exists():
                shutil.rmtree(latest_dir, ignore_errors=True)
            try:
                shutil.copytree(session_dir, latest_dir)
            except Exception:
                pass

            status_text.success(f"✅ Đã hoàn thành chẩn đoán và tự động lưu toàn bộ dữ liệu vào `output/`!")
            st.session_state["diagnosis_results"] = results
            st.session_state["saved_session_dir"] = str(session_dir.resolve())
            st.session_state["saved_gif_bytes"] = gif_bytes
            st.session_state["saved_gif_path"] = str(gif_file_path)

        results = st.session_state.get("diagnosis_results")

        if results:
            total_slices = len(results)
            normal_slices = [r for r in results if r["prediction"] == "Normal"]
            covid_slices = [r for r in results if r["prediction"] == "COVID-19"]
            normal_count = len(normal_slices)
            covid_count = len(covid_slices)
            avg_norm_prob = float(np.mean([r["probability_normal"] for r in results])) * 100
            avg_covid_prob = float(np.mean([r["probability_covid"] for r in results])) * 100
            avg_conf = float(np.mean([r["confidence"] for r in results]))

            saved_dir = st.session_state.get("saved_session_dir")
            if saved_dir:
                st.success(
                    f"💾 **Đã tự động lưu trữ toàn bộ ảnh tải lên và kết quả chẩn đoán vào ổ cứng:**\n\n"
                    f"- 📁 **Thư mục lưu:** `{saved_dir}` (và bản sao nhanh tại `output/latest/`)\n"
                    f"- 📥 **Ảnh gốc tải lên:** `{total_slices}` file tại thư mục con `uploads/`\n"
                    f"- 🎭 **Mặt nạ phổi U-Net:** `{total_slices}` file `_mask.png` tại `masks/`\n"
                    f"- 🎨 **Ảnh phủ tổn thương Cyan:** `{total_slices}` file `_overlay.png` tại `overlays/`\n"
                    f"- 🎞️ **Ảnh động Cine-Loop:** `patient_cineloop_{total_slices}_slices.gif`\n"
                    f"- 📊 **Bảng tổng hợp kết quả:** `batch_diagnosis_summary.csv`\n"
                    f"- 📝 **Báo cáo kết luận lâm sàng:** `clinical_report.txt`"
                )

                col_btn1, _ = st.columns([1, 3])
                with col_btn1:
                    if st.button("📂 Mở thư mục kết quả (Windows Explorer)"):
                        import subprocess
                        subprocess.Popen(f'explorer "{saved_dir}"')

            st.markdown("---")
            st.subheader("1. Kết Luận Lâm Sàng Tổng Thể")

            # Clinical Conclusion Banner formatted as requested
            if covid_count == 0:
                conclusion_text = f"Kết luận lâm sàng: Cả {total_slices} lát cắt của bệnh nhân đều được chẩn đoán là NORMAL (Âm tính với tổn thương đông đặc/kính mờ dạng COVID-19), với xác suất trung bình {avg_norm_prob:.1f}%."
                st.success(f"### 🩺 {conclusion_text}")
            else:
                conclusion_text = f"Kết luận lâm sàng: Phát hiện {covid_count}/{total_slices} lát cắt có dấu hiệu tổn thương nghi ngờ COVID-19 (Đông đặc/kính mờ), với xác suất trung bình {avg_covid_prob:.1f}%. Cần bác sĩ hội chẩn lâm sàng chi tiết!"
                st.error(f"### ⚠️ {conclusion_text}")

            # Metric KPI Cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Tổng số lát cắt", f"{total_slices}")
            kpi2.metric("Kết luận bệnh nhân", "NORMAL (Bình thường)" if covid_count == 0 else f"COVID-19 ({covid_count} lát)")
            kpi3.metric("Số lát Bình thường", f"{normal_count} / {total_slices} ({normal_count/total_slices*100:.0f}%)")
            kpi4.metric("Độ tin cậy TB", f"{avg_conf:.1f}%")

            # Visual Cine-Loop GIF Animation
            st.markdown("---")
            st.subheader("2. Ảnh Động Trực Quan Toàn Bộ Lát Cắt (Cine-Loop Animation)")
            gif_bytes = st.session_state.get("saved_gif_bytes")
            if not gif_bytes:
                with st.spinner("Đang tạo ảnh động Cine-Loop GIF từ các lát cắt CT..."):
                    gif_bytes = create_cine_gif(results, duration_ms=gif_duration, side_by_side=is_side_by_side)

            gif_col1, gif_col2 = st.columns([3, 1])
            with gif_col1:
                st.image(
                    gif_bytes,
                    caption=f"Ảnh động Cine-Loop CT Series ({total_slices} lát cắt | {gif_duration}ms/khung | {'Song song CT gốc & Lớp phủ' if is_side_by_side else 'Lớp phủ phân vùng'})",
                    use_column_width=True
                )
            with gif_col2:
                st.info(f"**Thông số Cine-Loop:**\n\n- **Số khung hình:** `{total_slices}`\n- **Tốc độ:** `{gif_duration} ms/lát` (~`{1000/gif_duration:.1f}` FPS)\n- **Chế độ:** `{'Song song' if is_side_by_side else 'Chỉ lớp phủ'}`\n- **Kích thước file:** `{len(gif_bytes)/1024:.1f} KB`")
                st.download_button(
                    label="📥 Tải xuống ảnh Cine-Loop GIF",
                    data=gif_bytes,
                    file_name=f"patient_cineloop_{total_slices}_slices.gif",
                    mime="image/gif",
                    use_container_width=True
                )

            # Summary Diagnosis Table
            st.markdown("---")
            st.subheader("3. Bảng Tổng Hợp Kết Quả Chẩn Đoán")

            summary_rows = []
            for r in results:
                summary_rows.append({
                    "Lát cắt": f"Slice #{r['slice_index']}",
                    "Tên file": r["filename"],
                    "Chẩn đoán": r["prediction"],
                    "Độ tin cậy": f"{r['confidence']:.2f}%",
                    "Xác suất Bình thường": f"{r['probability_normal']*100:.2f}%",
                    "Xác suất COVID-19": f"{r['probability_covid']*100:.2f}%"
                })
            summary_df = pd.DataFrame(summary_rows)

            st.dataframe(summary_df, use_container_width=True)

            csv_data = summary_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải Báo Cáo Tổng Hợp (CSV)",
                data=csv_data,
                file_name="bang_tong_hop_ket_qua_chan_doan.csv",
                mime="text/csv"
            )

            # Detailed Slice Explorer
            st.markdown("---")
            st.subheader("4. Khảo Sát Chi Tiết Từng Lát Cắt")

            sel_idx = st.selectbox(
                "Chọn lát cắt cụ thể để khảo sát hình ảnh phân vùng U-Net và 24 chỉ số cấu trúc Radiomics:",
                range(total_slices),
                format_func=lambda i: f"Lát #{results[i]['slice_index']}: {results[i]['filename']} [{results[i]['prediction']} - {results[i]['confidence']:.1f}%]"
            )

            sel_res = results[sel_idx]

            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                st.image(sel_res["image_gray"], caption=f"Ảnh CT Gốc ({sel_res['filename']})", use_column_width=True, clamp=True)
            with v_col2:
                st.image(sel_res["mask"], caption="Mặt nạ Nhu mô Phổi (U-Net Segmentation)", use_column_width=True, clamp=True)
            with v_col3:
                st.image(sel_res["overlay"], caption="Lớp phủ Tổn thương / Nhu mô (Cyan)", use_column_width=True)

            # Probability bar chart for selected slice
            prob_df = pd.DataFrame({
                "Tình trạng": ["Normal (Bình thường)", "COVID-19"],
                "Xác suất (%)": [sel_res["probability_normal"] * 100, sel_res["probability_covid"] * 100]
            })
            fig_prob = px.bar(
                prob_df, x="Tình trạng", y="Xác suất (%)", color="Tình trạng",
                color_discrete_map={"Normal (Bình thường)": "#28a745", "COVID-19": "#dc3545"},
                range_y=[0, 100], text_auto=".1f"
            )
            fig_prob.update_layout(height=260, showlegend=False, title=f"Phân phối Xác suất Lát #{sel_res['slice_index']} ({sel_res['filename']})")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Extracted Radiomics features
            st.write(f"**24 Đặc trưng Cấu trúc Radiomics trích xuất từ Lát #{sel_res['slice_index']}:**")
            feat_df = pd.DataFrame(list(sel_res["features"].items()), columns=["Chỉ số Sinh học Radiomics", "Giá trị tính toán"])
            st.dataframe(feat_df, use_container_width=True)

            # DICOM Scan Header Expander if present
            if sel_res.get("dicom_metadata"):
                with st.expander(f"📋 DICOM Header & Thông tin Y tế ({sel_res['filename']})"):
                    dcm_rows = [{"Thẻ DICOM": k, "Giá trị": str(v)} for k, v in sel_res["dicom_metadata"].items()]
                    st.table(pd.DataFrame(dcm_rows))

    
