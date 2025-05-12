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
import tempfile
import os
from pyulog import ULog

# Initialize session state for dataframes early to avoid key errors
if "b_df" not in st.session_state:
    st.session_state.b_df = None
if "v_df" not in st.session_state:
    st.session_state.v_df = None

st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")

# 🔹 Logo
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file {image_path} not found.")
        return ""

logo_base64 = get_base64_image("Rotrix-Logo.png")
# st.logo(logo_base64, *, size="medium", link=None, icon_image=None)
st.markdown(f"""
    <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
        <a href="http://rotrixdemo.reude.tech/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

# Loaders
def load_csv(file):
    try:
        file.seek(0)
        return pd.read_csv(StringIO(file.read().decode("utf-8")))
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def load_pcd(file):
    st.error("PCD file support is not available on this deployment. Please use CSV or ULOG files instead.")
    return pd.DataFrame()

def load_ulog(file, key_suffix=""):
    try:
        ulog = ULog(file)
        extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
        topic_names = ["None"] + list(extracted_dfs.keys())
        
        if not topic_names or len(topic_names) == 1:  # Only "None" in the list
            st.warning("No extractable topics found in ULOG file.")
            return pd.DataFrame()
        
        select_key = f"ulog_topic_{key_suffix}" if key_suffix else "ulog_topic"
        selected_topic = st.selectbox("Select a topic from extracted CSVs", topic_names, key=select_key)

        if selected_topic == "None":
            return pd.DataFrame()
            
        df = extracted_dfs.get(selected_topic, pd.DataFrame())
        if df.empty:
            st.warning(f"Topic `{selected_topic}` has no data.")
        
        return df
    except Exception as e:
        st.error(f"Error loading ULOG: {e}")
        return pd.DataFrame()

def detect_trend(series):
    if len(series) < 2:
        return "insufficient data"
    if series.iloc[-1] > series.iloc[0]:
        return "increasing"
    elif series.iloc[-1] < series.iloc[0]:
        return "decreasing"
    return "flat"

def detect_abnormalities(series, threshold=3.0):
    if len(series) < 2:
        return pd.Series([False] * len(series)), pd.Series([0] * len(series))
    
    # Avoid division by zero with a small epsilon
    std_dev = series.std()
    if std_dev == 0:
        return pd.Series([False] * len(series)), pd.Series([0] * len(series))
        
    z_scores = np.abs((series - series.mean()) / (std_dev + 1e-10))
    return z_scores > threshold, z_scores

def calculate_similarity_index(b_df, v_df, x_axis, y_axis):
    if x_axis == "None" or y_axis == "None":
        return 0
    
    merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))
    if merged.empty:
        return 0
        
    val_col = f"{y_axis}_validation"
    bench_col = f"{y_axis}_benchmark"
    
    rmse = np.sqrt(mean_squared_error(merged[bench_col], merged[val_col]))
    benchmark_range = merged[bench_col].max() - merged[bench_col].min()
    
    if benchmark_range > 0:
        similarity = 1 - (rmse / benchmark_range)
    else:
        similarity = 0
        
    return similarity * 100

def calculate_rmse(b_df, v_df, x_axis, y_axis):
    if x_axis == "None" or y_axis == "None":
        return 0
    
    merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))
    if merged.empty:
        return 0
        
    val_col = f"{y_axis}_validation"
    bench_col = f"{y_axis}_benchmark"
    
    return np.sqrt(mean_squared_error(merged[bench_col], merged[val_col]))

def calculate_abnormal_points(b_df, v_df, x_axis, y_axis, z_threshold):
    if x_axis == "None" or y_axis == "None":
        return 0
    
    merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))
    if merged.empty:
        return 0
        
    val_col = f"{y_axis}_validation"
    abnormal_mask, _ = detect_abnormalities(merged[val_col], z_threshold)
    return int(abnormal_mask.sum())

# Load logic
def load_data(file, filetype, key_suffix):
    if file is None:
        return None
        
    if filetype == ".csv":
        return load_csv(file)
    elif filetype == ".pcd":
        return load_pcd(file)
    elif filetype == ".ulg":
        return load_ulog(file, key_suffix)
    else:
        st.warning(f"Unsupported file type: {filetype}")
        return pd.DataFrame()

def add_remove_common_column(b_df, v_df):
    if b_df is None or v_df is None or b_df.empty or v_df.empty:
        st.warning("⚠️ Both Benchmark and Target data must be loaded.")
        return b_df, v_df

    if "pending_column" in st.session_state:
        new_col = st.session_state["pending_column"]
        try:
            for df_key in ["b_df", "v_df"]:
                df = st.session_state[df_key]
                if df is not None and not df.empty:
                    if new_col["name"] not in df.columns:
                        df.insert(1, new_col["name"], df.eval(new_col["formula"]))
                    else:
                        df[new_col["name"]] = df.eval(new_col["formula"])
                    st.session_state[df_key] = df
            st.success(f"✅ Added `{new_col['name']}` using `{new_col['formula']}` to both Benchmark and Target.")
        except Exception as e:
            st.error(f"❌ Failed to add column: {e}")
        del st.session_state["pending_column"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### 🧮 New Column")
        new_col_name = st.text_input("New Column Name", key="common_add")
        custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key="common_formula")

        if st.button("Add Column"):
            if new_col_name and custom_formula:
                st.session_state["pending_column"] = {
                    "name": new_col_name,
                    "formula": custom_formula
                }
                st.experimental_rerun()

    with col2:
        st.markdown("###### 🗑️ Remove Column")
        common_cols = []
        if b_df is not None and v_df is not None and not b_df.empty and not v_df.empty:
            common_cols = list(set(b_df.columns) & set(v_df.columns))
        cols_to_drop = st.multiselect("Select column(s) to drop", common_cols, key="common_drop")

        if st.button("Remove Columns"):
            if cols_to_drop:
                if b_df is not None and not b_df.empty:
                    st.session_state.b_df.drop(columns=[col for col in cols_to_drop if col in b_df.columns], inplace=True)
                if v_df is not None and not v_df.empty:
                    st.session_state.v_df.drop(columns=[col for col in cols_to_drop if col in v_df.columns], inplace=True)
                st.success(f"🗑️ Removed columns: {', '.join(cols_to_drop)} from both Benchmark and Target.")
                st.experimental_rerun()

    return st.session_state.b_df, st.session_state.v_df


def add_remove_column(target_df, df_name="DataFrame"):
    # CREATE COLUMN
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
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")

    # REMOVE COLUMN
    st.markdown("##### 🗑️ Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column from {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑️ Removed columns: {', '.join(columns_to_drop)} from {df_name}")
            
    st.markdown("##### ✏️ Rename Column")
    if not target_df.empty and len(target_df.columns) > 0:
        rename_col = st.selectbox("Select column to rename", target_df.columns, key=f"{df_name}_rename_col")
        new_name = st.text_input("New column name", key=f"{df_name}_rename_input")

        if st.button(f"Rename Column in {df_name}", key=f"{df_name}_rename_button"):
            if rename_col and new_name:
                target_df.rename(columns={rename_col: new_name}, inplace=True)
                st.success(f"✏️ Renamed column `{rename_col}` to `{new_name}` in {df_name}")

    return target_df

# Main content area
st.markdown("### Comparative Assessment")

# -----------------------------
# Modified sidebar & main layout
# -----------------------------
with st.sidebar:
    st.markdown("<h4 style='font-size:20px; color:#FFFF00;'>🔼 Upload Files</h4>", unsafe_allow_html=True)
    
    # Upload files section in sidebar
    st.markdown("#### 📂 Upload Benchmark File")
    benchmark_files = st.file_uploader("Benchmark Files", type=["csv", "ulg"], accept_multiple_files=True, 
                                      label_visibility="collapsed")
    benchmark_names = [f.name for f in benchmark_files] if benchmark_files else []
    
    st.markdown("#### 📂 Upload Target File")
    validation_files = st.file_uploader("Target Files", type=["csv", "ulg"], accept_multiple_files=True,
                                      label_visibility="collapsed")
    validation_names = [f.name for f in validation_files] if validation_files else []
    
    # MOVED: Parameters section to sidebar
    st.markdown("<h4 style='font-size:18px; color:#0099ff;'>📈 Parameters</h4>", unsafe_allow_html=True)
    
    # Get current dataframes from session state for column extraction
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")
    
    # Find common columns for dropdowns
    common_cols = []
    if b_df is not None and v_df is not None and not b_df.empty and not v_df.empty:
        common_cols = list(set(b_df.columns) & set(v_df.columns))
    
    # Parameter controls in sidebar
    x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_select")
    y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_select")
    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider")
    
    if x_axis == "None" or y_axis == "None":
        st.info("📌 Please select both X-axis and Y-axis to compare.")
    
    # Data Analysis Settings moved to sidebar
    st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
    selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key='data_analysis')

    for param in selected_df:
        if param == "Both":
            st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)

        elif param == "Benchmark":
            st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")

        elif param == "Target":
            st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")

# MAIN CONTENT AREA
# File selection moved to main content in two columns
col_select1, col_select2 = st.columns(2)

with col_select1:
    st.markdown("#### 🧪 Select Benchmark File")
    selected_bench = st.selectbox("Select Benchmark File", ["None"] + benchmark_names)
    if selected_bench != "None" and benchmark_files:
        b_file = benchmark_files[benchmark_names.index(selected_bench)]
        b_file_ext = os.path.splitext(b_file.name)[-1].lower()
        st.session_state.b_df = load_data(b_file, b_file_ext, key_suffix="bench")

with col_select2:
    st.markdown("#### 🔬 Select Target File")
    selected_val = st.selectbox("Select Target File", ["None"] + validation_names)
    if selected_val != "None" and validation_files:
        v_file = validation_files[validation_names.index(selected_val)]
        v_file_ext = os.path.splitext(v_file.name)[-1].lower()
        st.session_state.v_df = load_data(v_file, v_file_ext, key_suffix="val")

# KPI Cards in 4 columns
if b_df is not None and v_df is not None and not b_df.empty and not v_df.empty:
    st.markdown("### 📊 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        try:
            similarity_index = calculate_similarity_index(b_df, v_df, x_axis, y_axis)
            st.metric("Similarity Index", f"{similarity_index:.2f}%")
        except:
            st.metric("Similarity Index", "N/A")
    
    with kpi_col2:
        try:
            rmse = calculate_rmse(b_df, v_df, x_axis, y_axis)
            st.metric("RMSE", f"{rmse:.2f}")
        except:
            st.metric("RMSE", "N/A")
    
    with kpi_col3:
        try:
            abnormal_points = calculate_abnormal_points(b_df, v_df, x_axis, y_axis, z_threshold)
            st.metric("Abnormal Points", f"{abnormal_points}")
        except:
            st.metric("Abnormal Points", "N/A")
    
    with kpi_col4:
        try:
            trend = detect_trend(b_df[y_axis])
            st.metric("Trend", trend)
        except:
            st.metric("Trend", "N/A")

# Main content tabs
tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])

with tab2:
    st.subheader("📁 Imported Data Preview")
    
    # Get updated dataframes from session state
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")
    
    if b_df is not None and v_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧪 Benchmark Data")
            st.dataframe(b_df)
        with col2:
            st.markdown("### 🔬 Target Data")
            st.dataframe(v_df)

    elif b_df is not None:
        st.markdown("### 🧪 Benchmark Data")
        st.dataframe(b_df)

    elif v_df is not None:
        st.markdown("### 🔬 Target Data")
        st.dataframe(v_df)

    else:
        st.info("No data uploaded yet.")

with tab1:
    st.subheader(" 🔍 Comparative Analysis")
    b_df = st.session_state.get("b_df") 
    v_df = st.session_state.get("v_df")
    
    if b_df is not None and v_df is not None and not b_df.empty and not v_df.empty:
        # Add index column if it doesn't exist
        if "Index" not in b_df.columns:
            b_df.insert(0, "Index", range(1, len(b_df) + 1))
            st.session_state.b_df = b_df
        if "Index" not in v_df.columns:
            v_df.insert(0, "Index", range(1, len(v_df) + 1))
            st.session_state.v_df = v_df

        common_cols = list(set(b_df.columns) & set(v_df.columns))
        if common_cols:
            # Get x_axis and y_axis from session state or sidebar selections
            x_axis = st.session_state.get("x_axis_select", "None")
            y_axis = st.session_state.get("y_axis_select", "None")
            z_threshold = st.session_state.get("z-slider", 3.0)
            
            if x_axis == "None" or y_axis == "None":
                st.info("📌 Please select both X-axis and Y-axis in the sidebar to compare.")
            else:
                # Plot & metrics in columns
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    # Safe handling of min/max values
                    try:
                        x_min = float(b_df[x_axis].min())
                        x_max = float(b_df[x_axis].max())
                        y_min = float(b_df[y_axis].min())
                        y_max = float(b_df[y_axis].max())

                        # Filter data safely
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                        (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                        (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                        
                        # Check if filtered data is not empty
                        if b_filtered.empty or v_filtered.empty:
                            st.warning("No data points after filtering. Please adjust filter values.")
                            merged = pd.DataFrame()
                            abnormal_mask = pd.Series()
                            z_scores = pd.Series()
                            abnormal_points = pd.DataFrame()
                        else:
                            merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'))
                            
                            if merged.empty:
                                st.warning("No common points between datasets after filtering.")
                                abnormal_mask = pd.Series()
                                z_scores = pd.Series()
                                abnormal_points = pd.DataFrame()
                            else:
                                val_col = f"{y_axis}_validation"
                                bench_col = f"{y_axis}_benchmark"
                                abnormal_mask, z_scores = detect_abnormalities(merged[val_col], z_threshold)
                                merged["Z_Score"] = z_scores
                                merged["Abnormal"] = abnormal_mask
                                abnormal_points = merged[merged["Abnormal"]]
                                
                                # Plot layout
                                st.markdown("### 🧮 Plot Visualization")
                                fig = make_subplots(rows=2, cols=1, subplot_titles=["Benchmark", "Target"], shared_yaxes=True)
    
                                fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_benchmark"], 
                                                        name="Benchmark", mode="lines"), row=1, col=1)
                                fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_validation"], 
                                                        name="Target", mode="lines"), row=2, col=1)
                                
                                if not abnormal_points.empty:
                                    fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                                            mode='markers', marker=dict(color='red', size=6),
                                                            name="Abnormal"), row=2, col=1)
    
                                fig.update_layout(height=700, title_text="Benchmark vs Target Subplot")
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error in data processing: {e}")
                        merged = pd.DataFrame()
                        abnormal_mask = pd.Series()
                        z_scores = pd.Series()
                        abnormal_points = pd.DataFrame()

                with col2:
                    if x_axis == "None" or y_axis == "None":
                        st.info("📌 Please select both X-axis and Y-axis in the sidebar to compare.")
                    elif merged.empty:
                        st.warning("No data available for metrics calculation.")
                    # Removed metrics gauge/indicator block here

        else:
            st.warning("No common columns to compare between Benchmark and Target datasets.")
    else:
        st.info("Please upload both benchmark and target files for comparison.")
