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
    st.title("🩺 Clinical Diagnosis on New Patient Images")
    st.write("Upload a chest CT scan / X-ray image to automatically segment the lung region (U-Net), extract radiomic texture features, and predict COVID-19 vs. Normal classification.")

    from modules.inference import diagnose_image
    from PIL import Image

    uploaded_file = st.file_uploader("Choose a Chest CT / X-ray image (JPG, PNG, JPEG, DICOM .dcm):", type=["jpg", "jpeg", "png", "dcm", "dicom"])

    if uploaded_file is not None:
        import io
        from modules.inference import load_dicom_image, is_dicom_input
        file_bytes = uploaded_file.getvalue()
        is_dcm = uploaded_file.name.lower().endswith(('.dcm', '.dicom')) or is_dicom_input(io.BytesIO(file_bytes))
        dcm_meta = {}

        col_img, col_info = st.columns([1, 2])
        with col_img:
            if is_dcm:
                try:
                    img_arr, dcm_meta = load_dicom_image(io.BytesIO(file_bytes))
                    preview_img = Image.fromarray(img_arr)
                    st.image(preview_img, caption=f"DICOM CT Scan Preview ({uploaded_file.name})", width=250)
                except Exception as dcm_e:
                    st.error(f"Error reading DICOM: {dcm_e}")
                    preview_img = None
            else:
                preview_img = Image.open(io.BytesIO(file_bytes))
                st.image(preview_img, caption=f"Scan Preview ({uploaded_file.name})", width=250)

        with col_info:
            if is_dcm and dcm_meta:
                st.success(f"**DICOM Detected**: Modality: `{dcm_meta.get('Modality', 'CT')}` | WL/WW: `{dcm_meta.get('WindowCenter', -600)}/{dcm_meta.get('WindowWidth', 1500)} HU`")
            st.info("Click the button below to execute the end-to-end pipeline (U-Net Segmentation -> 24 PyRadiomics Feature Extraction -> Random Forest Classification).")
            run_diag = st.button("🚀 Run Clinical Diagnosis", type="primary", use_container_width=True)

        if run_diag and preview_img is not None:
            with st.spinner("Processing scan: segmenting lung parenchyma and extracting radiomic biomarkers..."):
                result = diagnose_image(io.BytesIO(file_bytes))

            pred = result["prediction"]
            conf = result["confidence"]
            p_covid = result["probability_covid"] * 100
            p_norm = result["probability_normal"] * 100

            st.markdown("---")
            st.subheader("1. Diagnostic Assessment")

            m_col1, m_col2, m_col3 = st.columns(3)
            if pred == "COVID-19":
                m_col1.error(f"### Predicted: {pred}")
            else:
                m_col1.success(f"### Predicted: {pred}")

            m_col2.metric("Confidence Level", f"{conf:.1f}%")
            m_col3.metric("COVID-19 Risk Probability", f"{p_covid:.1f}%", delta=f"{p_covid - 50:.1f}%", delta_color="inverse")

            # Probability Comparison Bar
            prob_df = pd.DataFrame({
                "Condition": ["Normal", "COVID-19"],
                "Probability (%)": [p_norm, p_covid]
            })
            fig_prob = px.bar(
                prob_df, x="Condition", y="Probability (%)", color="Condition",
                color_discrete_map={"Normal": "#28a745", "COVID-19": "#dc3545"},
                range_y=[0, 100], text_auto=".1f"
            )
            fig_prob.update_layout(height=280, showlegend=False, title="Prediction Probability Distribution")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Visualizations
            st.subheader("2. Segmentation & Lesion Inspection")
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                st.image(result["image_gray"], caption="Original Input CT Scan", use_container_width=True, clamp=True)
            with v_col2:
                st.image(result["mask"], caption="Predicted Lung Mask (U-Net)", use_container_width=True, clamp=True)
            with v_col3:
                st.image(result["overlay"], caption="Parenchyma Mask Overlay (Cyan)", use_container_width=True)

            # Extracted Radiomics Features
            st.subheader("3. Extracted Radiomic Texture Biomarkers (24 Features)")
            feat_df = pd.DataFrame(list(result["features"].items()), columns=["Radiomic Feature", "Computed Value"])
            st.dataframe(feat_df, use_container_width=True)

            # Optional DICOM Metadata Expander
            if result.get("dicom_metadata"):
                with st.expander("📋 DICOM Scan Header & Clinical Metadata"):
                    dcm_rows = [{"DICOM Tag": k, "Value": str(v)} for k, v in result["dicom_metadata"].items()]
                    st.table(pd.DataFrame(dcm_rows))

    
