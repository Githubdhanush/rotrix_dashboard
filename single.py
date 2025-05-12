# Modified script to support both single file analysis and comparative assessment modes

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from io import StringIO
from PIL import Image
import base64
# import open3d as o3d
import tempfile
import os
from pyulog import ULog

# Initialize session state variables
if 'single_df' not in st.session_state:
    st.session_state.single_df = None
if 'b_df' not in st.session_state:
    st.session_state.b_df = None
if 'v_df' not in st.session_state:
    st.session_state.v_df = None
if 'selected_bench' not in st.session_state:
    st.session_state.selected_bench = "None"
if 'selected_val' not in st.session_state:
    st.session_state.selected_val = "None"
if 'benchmark_data' not in st.session_state:
    st.session_state.benchmark_data = {}
if 'validation_data' not in st.session_state:
    st.session_state.validation_data = {}

# --- Helper functions (move all here) ---
def convert_numeric_columns(df):
    if df is None or df.empty:
        return df
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            continue
    return df

def load_csv(file):
    file.seek(0)
    df = pd.read_csv(StringIO(file.read().decode("utf-8")))
    return convert_numeric_columns(df)

# def load_pcd(file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
#         tmp.write(file.read())
#         pcd = o3d.io.read_point_cloud(tmp.name, format='xyz')
#         points = np.asarray(pcd.points)
#         df = pd.DataFrame(points, columns=["X", "Y", "Z"])
#         if len(np.asarray(pcd.colors)) > 0:
#             df["Temperature"] = np.mean(np.asarray(pcd.colors), axis=1)
#     return df

def load_ulog(file, key_suffix=""):
    ulog = ULog(file)
    extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
    topic_names = ["None"] + list(extracted_dfs.keys())
    if not topic_names:
        st.warning("No extractable topics found in ULOG file.")
        return pd.DataFrame()
    select_key = f"ulog_topic_{key_suffix}" if key_suffix else None
    selected_topic = st.selectbox("Select a topic from extracted CSVs", topic_names, key=select_key)
    df = extracted_dfs.get(selected_topic, pd.DataFrame())
    if df.empty:
        st.warning(f"Topic `{selected_topic}` has no data.")
    else:
        df = convert_numeric_columns(df)
    return df

def load_data(file, filetype, key_suffix):
    if filetype == ".csv":
        df_csv = load_csv(file)
        return df_csv
    elif filetype == ".pcd":
        df_pcd = load_pcd(file)
        return df_pcd
    elif filetype == ".ulg":
        df_ulog = load_ulog(file, key_suffix)
        return df_ulog
    return None

def detect_trend(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "Cannot determine trend for non-numeric data"
    if series.iloc[-1] > series.iloc[0]:
        return "increasing"
    elif series.iloc[-1] < series.iloc[0]:
        return "decreasing"
    return "flat"

def detect_abnormalities(series, threshold=3.0):
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

def add_remove_column(target_df, df_name="DataFrame"):
    if target_df is None or target_df.empty:
        st.warning(f"⚠️ {df_name} is empty or not loaded.")
        return target_df
    st.markdown("##### 🧮 New Column")
    new_col_name = st.text_input("New Column Name", key=f"{df_name}_add")
    custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key=f"{df_name}_formula")
    if st.button(f"Add Column to {df_name}"):
        try:
            if new_col_name and custom_formula:
                target_df[new_col_name] = target_df.eval(custom_formula)
                st.success(f"✅ Added column `{new_col_name}` to {df_name} using: `{custom_formula}`")
                if df_name == "Data":
                    st.session_state.single_df = target_df.copy()
                elif df_name == "Benchmark":
                    st.session_state.b_df = target_df.copy()
                    if st.session_state.selected_bench != "None":
                        st.session_state.benchmark_data[st.session_state.selected_bench] = target_df.copy()
                elif df_name == "Target":
                    st.session_state.v_df = target_df.copy()
                    if st.session_state.selected_val != "None":
                        st.session_state.validation_data[st.session_state.selected_val] = target_df.copy()
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")
    st.markdown("##### 🗑️ Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")
    if st.button(f"Remove Column from {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑️ Removed columns: {', '.join(columns_to_drop)} from {df_name}")
            if df_name == "Data":
                st.session_state.single_df = target_df.copy()
            elif df_name == "Benchmark":
                st.session_state.b_df = target_df.copy()
                if st.session_state.selected_bench != "None":
                    st.session_state.benchmark_data[st.session_state.selected_bench] = target_df.copy()
            elif df_name == "Target":
                st.session_state.v_df = target_df.copy()
                if st.session_state.selected_val != "None":
                    st.session_state.validation_data[st.session_state.selected_val] = target_df.copy()
    st.markdown("##### ✏️ Rename Column")
    rename_col = st.selectbox("Select column to rename", target_df.columns, key=f"{df_name}_rename_col")
    new_name = st.text_input("New column name", key=f"{df_name}_rename_input")
    if st.button(f"Rename Column in {df_name}", key=f"{df_name}_rename_button"):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"✏️ Renamed column `{rename_col}` to `{new_name}` in {df_name}")
            if df_name == "Data":
                st.session_state.single_df = target_df.copy()
            elif df_name == "Benchmark":
                st.session_state.b_df = target_df.copy()
                if st.session_state.selected_bench != "None":
                    st.session_state.benchmark_data[st.session_state.selected_bench] = target_df.copy()
            elif df_name == "Target":
                st.session_state.v_df = target_df.copy()
                if st.session_state.selected_val != "None":
                    st.session_state.validation_data[st.session_state.selected_val] = target_df.copy()
    return target_df

def add_remove_common_column(b_df, v_df):
    if b_df is None or v_df is None or b_df.empty or v_df.empty:
        st.warning("⚠️ Both DataFrames must be loaded to modify common columns.")
        return b_df, v_df
    st.markdown("##### 🧮 Common Column Operations")
    common_cols = list(set(b_df.columns) & set(v_df.columns))
    if not common_cols:
        st.warning("No common columns found between DataFrames.")
        return b_df, v_df
    operation = st.selectbox("Select Operation", ["Add", "Remove", "Rename"], key="common_op")
    if operation == "Add":
        new_col_name = st.text_input("New Column Name", key="common_add")
        custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key="common_formula")
        if st.button("Add Column to Both"):
            try:
                if new_col_name and custom_formula:
                    b_df[new_col_name] = b_df.eval(custom_formula)
                    v_df[new_col_name] = v_df.eval(custom_formula)
                    st.success(f"✅ Added column `{new_col_name}` to both DataFrames")
                    st.session_state.b_df = b_df.copy()
                    st.session_state.v_df = v_df.copy()
                    if st.session_state.selected_bench != "None":
                        st.session_state.benchmark_data[st.session_state.selected_bench] = b_df.copy()
                    if st.session_state.selected_val != "None":
                        st.session_state.validation_data[st.session_state.selected_val] = v_df.copy()
            except Exception as e:
                st.error(f"❌ Error creating column: {e}")
    elif operation == "Remove":
        cols_to_drop = st.multiselect("Select columns to remove", common_cols, key="common_drop")
        if st.button("Remove from Both"):
            if cols_to_drop:
                b_df.drop(columns=cols_to_drop, inplace=True)
                v_df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"🗑️ Removed columns: {', '.join(cols_to_drop)} from both DataFrames")
                st.session_state.b_df = b_df.copy()
                st.session_state.v_df = v_df.copy()
                if st.session_state.selected_bench != "None":
                    st.session_state.benchmark_data[st.session_state.selected_bench] = b_df.copy()
                if st.session_state.selected_val != "None":
                    st.session_state.validation_data[st.session_state.selected_val] = v_df.copy()
    elif operation == "Rename":
        rename_col = st.selectbox("Select column to rename", common_cols, key="common_rename_col")
        new_name = st.text_input("New column name", key="common_rename_input")
        if st.button("Rename in Both"):
            if rename_col and new_name:
                b_df.rename(columns={rename_col: new_name}, inplace=True)
                v_df.rename(columns={rename_col: new_name}, inplace=True)
                st.success(f"✏️ Renamed column `{rename_col}` to `{new_name}` in both DataFrames")
                st.session_state.b_df = b_df.copy()
                st.session_state.v_df = v_df.copy()
                if st.session_state.selected_bench != "None":
                    st.session_state.benchmark_data[st.session_state.selected_bench] = b_df.copy()
                if st.session_state.selected_val != "None":
                    st.session_state.validation_data[st.session_state.selected_val] = v_df.copy()
    return b_df, v_df
# --- End helper functions ---

st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")

# 🔹 Logo
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning("Logo file not found. Running without logo.")
        return None

logo_base64 = get_base64_image("Rotrix-Logo.png")
if logo_base64:
    st.markdown(f"""
        <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
            <a href="http://rotrixdemo.reude.tech/" target="_blank">
                <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
            </a>
        </div>
    """, unsafe_allow_html=True)

st.markdown("### 🚀 ROTRIX Data Analysis")

# File Upload Section
st.markdown("<h4 style='font-size:20px; color:#FFFF00;'>🔼 Upload Files</h4>", unsafe_allow_html=True)

top_col1, top_col2 = st.columns(2)

with top_col1:
    benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    if benchmark_files:
        for file in benchmark_files:
            if file.name not in st.session_state.benchmark_data:
                file_ext = os.path.splitext(file.name)[-1].lower()
                df = load_data(file, file_ext, key_suffix="bench")
                if df is not None and not df.empty:
                    st.session_state.benchmark_data[file.name] = df
    benchmark_names = list(st.session_state.benchmark_data.keys())
    if benchmark_names:
        selected_bench = st.selectbox("Select Benchmark File", ["None"] + benchmark_names, 
                                    index=0 if st.session_state.selected_bench not in benchmark_names 
                                    else benchmark_names.index(st.session_state.selected_bench) + 1,
                                    key="bench_select")
        if selected_bench != "None":
            st.session_state.selected_bench = selected_bench
            st.session_state.b_df = st.session_state.benchmark_data[selected_bench]

with top_col2:
    validation_files = st.file_uploader("📂 Upload Target File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    if validation_files:
        for file in validation_files:
            if file.name not in st.session_state.validation_data:
                file_ext = os.path.splitext(file.name)[-1].lower()
                df = load_data(file, file_ext, key_suffix="val")
                if df is not None and not df.empty:
                    st.session_state.validation_data[file.name] = df
    validation_names = list(st.session_state.validation_data.keys())
    if validation_names:
        selected_val = st.selectbox("Select Target File", ["None"] + validation_names,
                                  index=0 if st.session_state.selected_val not in validation_names 
                                  else validation_names.index(st.session_state.selected_val) + 1,
                                  key="val_select")
        if selected_val != "None":
            st.session_state.selected_val = selected_val
            st.session_state.v_df = st.session_state.validation_data[selected_val]

# Determine analysis mode based on uploaded files
if st.session_state.b_df is not None and st.session_state.v_df is not None:
    # Comparative Analysis Mode
    st.markdown("### 🔄 Comparative Analysis Mode")
    
    # Data Analysis Section
    col_main1, col_main2 = st.columns([0.25, 0.75])
    
    with col_main1:
        st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
        selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key='data_analysis')
        
        for param in selected_df:
            if param == "Both":
                st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
            elif param == "Benchmark":
                st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")
            elif param == "Target":
                st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")
    
    with col_main2:
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        with tab2:
            st.subheader("📁 Imported Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🧪 Benchmark Data")
                st.dataframe(st.session_state.b_df)
            with col2:
                st.markdown("### 🔬 Target Data")
                st.dataframe(st.session_state.v_df)
        
        with tab1:
            st.subheader("🔍 Comparative Analysis")
            
            if st.session_state.b_df is not None and st.session_state.v_df is not None:
                b_df = st.session_state.b_df.copy()
                v_df = st.session_state.v_df.copy()
                
                # Get numeric columns only
                b_numeric_cols = b_df.select_dtypes(include=[np.number]).columns.tolist()
                v_numeric_cols = v_df.select_dtypes(include=[np.number]).columns.tolist()
                common_cols = list(set(b_numeric_cols) & set(v_numeric_cols))
                
                if common_cols:
                    col1, col2, col3 = st.columns([0.20, 0.60, 0.20])
                    with col1:
                        st.markdown("#### 📈 Parameters")
                        x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_select")
                        y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_select")
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider")
                        
                        if x_axis != "None" and y_axis != "None":
                            try:
                                x_min = st.number_input("X min", value=float(b_df[x_axis].min()), min_value=float(b_df[x_axis].min()), max_value=float(b_df[x_axis].max()), key="comp_x_min")
                                x_max = st.number_input("X max", value=float(b_df[x_axis].max()), min_value=float(b_df[x_axis].min()), max_value=float(b_df[x_axis].max()), key="comp_x_max")
                                y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), min_value=float(b_df[y_axis].min()), max_value=float(b_df[y_axis].max()), key="comp_y_min")
                                y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), min_value=float(b_df[y_axis].min()), max_value=float(b_df[y_axis].max()), key="comp_y_max")
                                
                                # Filter data
                                b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                                (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                                v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                                (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                                
                                merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'))
                                
                                if merged.empty:
                                    st.warning("No overlapping data after filtering. Please adjust your X/Y min/max or check your data.")
                                else:
                                    val_col = f"{y_axis}_validation"
                                    bench_col = f"{y_axis}_benchmark"
                                    abnormal_mask, z_scores = detect_abnormalities(merged[val_col], z_threshold)
                                    merged["Z_Score"] = z_scores
                                    merged["Abnormal"] = abnormal_mask
                                    abnormal_points = merged[merged["Abnormal"]]
                                    
                                    with col2:
                                        st.markdown("### 🧮 Plot Visualization")
                                        fig = go.Figure()
                                        fig = make_subplots(rows=2, cols=1, subplot_titles=["Benchmark", "Target"], shared_yaxes=True)
                                        
                                        fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_benchmark"], name="Benchmark", mode="lines"), row=1, col=1)
                                        fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_validation"], name="Target", mode="lines"), row=2, col=1)
                                        fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                                             mode='markers', marker=dict(color='red', size=6),
                                                             name="Abnormal"), row=2, col=1)
                                        
                                        fig.update_layout(height=700, width=1000, title_text="Benchmark vs Target Subplot")
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col3:
                                        st.markdown("### 🎯 Metrics")
                                        rmse = np.sqrt(mean_squared_error(merged[bench_col], merged[val_col]))
                                        similarity = 1 - (rmse / (merged[bench_col].max() - merged[bench_col].min()))
                                        similarity_index = similarity * 100
                                        
                                        fig = make_subplots(
                                            rows=3, cols=1,
                                            specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
                                            vertical_spacing=0.05
                                        )
                                        
                                        fig.add_trace(go.Indicator(
                                            mode="gauge+number+delta",
                                            value=similarity_index,
                                            title={'text': "Similarity Index (%)"},
                                            delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                            gauge={
                                                'axis': {'range': [0, 100]},
                                                'bar': {'color': "darkblue"},
                                                'steps': [
                                                    {'range': [0, 50], 'color': "red"},
                                                    {'range': [50, 75], 'color': "orange"},
                                                    {'range': [75, 100], 'color': "green"}
                                                ],
                                                'threshold': {
                                                    'line': {'color': "black", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': similarity_index
                                                }
                                            }
                                        ), row=1, col=1)
                                        
                                        fig.add_trace(go.Indicator(
                                            mode="gauge+number",
                                            value=rmse,
                                            title={'text': "RMSE Error"},
                                            gauge={
                                                'axis': {'range': [0, max(100, rmse * 2)]},
                                                'bar': {'color': "orange"},
                                                'steps': [
                                                    {'range': [0, 10], 'color': "#d4f0ff"},
                                                    {'range': [10, 30], 'color': "#ffeaa7"},
                                                    {'range': [30, 100], 'color': "#ff7675"}
                                                ]
                                            }
                                        ), row=2, col=1)
                                        
                                        fig.add_trace(go.Indicator(
                                            mode="gauge+number",
                                            value=abnormal_mask.sum(),
                                            title={'text': "Abnormal Points"},
                                            gauge={
                                                'axis': {'range': [0, max(10, abnormal_mask.sum() * 2)]},
                                                'bar': {'color': "crimson"},
                                                'steps': [
                                                    {'range': [0, 10], 'color': "#c8e6c9"},
                                                    {'range': [10, 25], 'color': "#ffcc80"},
                                                    {'range': [25, 100], 'color': "#ef5350"}
                                                ]
                                            }
                                        ), row=3, col=1)
                                        
                                        fig.update_layout(height=700, margin=dict(t=10, b=10))
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error in analysis: {str(e)}")
                else:
                    st.warning("No common numeric columns to compare between Benchmark and Validation.")
            else:
                st.info("Please upload both benchmark and validation files.")
        
elif st.session_state.b_df is not None:
    # Single File Analysis Mode (Benchmark)
    st.markdown("### 📊 Single File Analysis Mode (Benchmark)")
    
    # Data Analysis Section
    col_main1, col_main2 = st.columns([0.25, 0.75])
    
    with col_main1:
        st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
        st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")
    
    with col_main2:
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        with tab2:
            st.subheader("📁 Data Preview")
            st.dataframe(st.session_state.b_df)
        
        with tab1:
            st.subheader("🔍 Data Visualization")
            numeric_columns = st.session_state.b_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                st.warning("No numeric columns found for visualization")
            else:
                x_axis = st.selectbox("Select X-Axis", numeric_columns, key="bench_x_axis")
                y_axis = st.selectbox("Select Y-Axis", [col for col in numeric_columns if col != x_axis], key="bench_y_axis")
                
                if x_axis and y_axis:
                    # Basic statistics
                    st.markdown("### 📊 Basic Statistics")
                    stats = st.session_state.b_df[y_axis].describe()
                    st.write(stats)
                    
                    # User-adjustable min/max for plot
                    x_min = st.number_input("X min", value=float(st.session_state.b_df[x_axis].min()), min_value=float(st.session_state.b_df[x_axis].min()), max_value=float(st.session_state.b_df[x_axis].max()), key="single_x_min")
                    x_max = st.number_input("X max", value=float(st.session_state.b_df[x_axis].max()), min_value=float(st.session_state.b_df[x_axis].min()), max_value=float(st.session_state.b_df[x_axis].max()), key="single_x_max")
                    y_min = st.number_input("Y min", value=float(st.session_state.b_df[y_axis].min()), min_value=float(st.session_state.b_df[y_axis].min()), max_value=float(st.session_state.b_df[y_axis].max()), key="single_y_min")
                    y_max = st.number_input("Y max", value=float(st.session_state.b_df[y_axis].max()), min_value=float(st.session_state.b_df[y_axis].min()), max_value=float(st.session_state.b_df[y_axis].max()), key="single_y_max")

                    filtered_df = st.session_state.b_df[(st.session_state.b_df[x_axis] >= x_min) & (st.session_state.b_df[x_axis] <= x_max) & (st.session_state.b_df[y_axis] >= y_min) & (st.session_state.b_df[y_axis] <= y_max)]

                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=filtered_df[x_axis], y=filtered_df[y_axis], mode="lines", name="Data"))
                    try:
                        z = np.polyfit(range(len(filtered_df)), filtered_df[y_axis], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(x=filtered_df[x_axis], y=p(range(len(filtered_df))), mode="lines", name="Trend Line", line=dict(dash="dash")))
                    except Exception as e:
                        st.warning(f"Could not generate trend line: {str(e)}")
                    fig.update_layout(title=f"{y_axis} vs {x_axis}", xaxis_title=x_axis, yaxis_title=y_axis, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend analysis
                    trend = detect_trend(st.session_state.b_df[y_axis])
                    st.markdown(f"### 📈 Trend Analysis")
                    st.markdown(f"- Overall trend: **{trend}**")
                    
                    # Abnormalities detection
                    st.markdown("### ⚠️ Abnormalities Detection")
                    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="bench_z_slider")
                    abnormal_mask, z_scores = detect_abnormalities(st.session_state.b_df[y_axis], z_threshold)
                    abnormal_points = st.session_state.b_df[abnormal_mask]
                    
                    if not abnormal_points.empty:
                        st.markdown(f"Found {len(abnormal_points)} abnormal points")
                        st.dataframe(abnormal_points[[x_axis, y_axis]])
                    else:
                        st.success("No abnormalities detected")

elif st.session_state.v_df is not None:
    # Single File Analysis Mode (Target)
    st.markdown("### 📊 Single File Analysis Mode (Target)")
    
    # Data Analysis Section
    col_main1, col_main2 = st.columns([0.25, 0.75])
    
    with col_main1:
        st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
        st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")
    
    with col_main2:
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        with tab2:
            st.subheader("📁 Data Preview")
            st.dataframe(st.session_state.v_df)
        
        with tab1:
            st.subheader("🔍 Data Visualization")
            numeric_columns = st.session_state.v_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                st.warning("No numeric columns found for visualization")
            else:
                x_axis = st.selectbox("Select X-Axis", numeric_columns, key="val_x_axis")
                y_axis = st.selectbox("Select Y-Axis", [col for col in numeric_columns if col != x_axis], key="val_y_axis")
                
                if x_axis and y_axis:
                    # Basic statistics
                    st.markdown("### 📊 Basic Statistics")
                    stats = st.session_state.v_df[y_axis].describe()
                    st.write(stats)
                    
                    # User-adjustable min/max for plot
                    x_min = st.number_input("X min", value=float(st.session_state.v_df[x_axis].min()), min_value=float(st.session_state.v_df[x_axis].min()), max_value=float(st.session_state.v_df[x_axis].max()), key="single_x_min")
                    x_max = st.number_input("X max", value=float(st.session_state.v_df[x_axis].max()), min_value=float(st.session_state.v_df[x_axis].min()), max_value=float(st.session_state.v_df[x_axis].max()), key="single_x_max")
                    y_min = st.number_input("Y min", value=float(st.session_state.v_df[y_axis].min()), min_value=float(st.session_state.v_df[y_axis].min()), max_value=float(st.session_state.v_df[y_axis].max()), key="single_y_min")
                    y_max = st.number_input("Y max", value=float(st.session_state.v_df[y_axis].max()), min_value=float(st.session_state.v_df[y_axis].min()), max_value=float(st.session_state.v_df[y_axis].max()), key="single_y_max")

                    filtered_df = st.session_state.v_df[(st.session_state.v_df[x_axis] >= x_min) & (st.session_state.v_df[x_axis] <= x_max) & (st.session_state.v_df[y_axis] >= y_min) & (st.session_state.v_df[y_axis] <= y_max)]

                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=filtered_df[x_axis], y=filtered_df[y_axis], mode="lines", name="Data"))
                    try:
                        z = np.polyfit(range(len(filtered_df)), filtered_df[y_axis], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(x=filtered_df[x_axis], y=p(range(len(filtered_df))), mode="lines", name="Trend Line", line=dict(dash="dash")))
                    except Exception as e:
                        st.warning(f"Could not generate trend line: {str(e)}")
                    fig.update_layout(title=f"{y_axis} vs {x_axis}", xaxis_title=x_axis, yaxis_title=y_axis, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend analysis
                    trend = detect_trend(st.session_state.v_df[y_axis])
                    st.markdown(f"### 📈 Trend Analysis")
                    st.markdown(f"- Overall trend: **{trend}**")
                    
                    # Abnormalities detection
                    st.markdown("### ⚠️ Abnormalities Detection")
                    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="val_z_slider")
                    abnormal_mask, z_scores = detect_abnormalities(st.session_state.v_df[y_axis], z_threshold)
                    abnormal_points = st.session_state.v_df[abnormal_mask]
                    
                    if not abnormal_points.empty:
                        st.markdown(f"Found {len(abnormal_points)} abnormal points")
                        st.dataframe(abnormal_points[[x_axis, y_axis]])
                    else:
                        st.success("No abnormalities detected")

else:
    st.info("Please upload at least one file to begin analysis.")

        
