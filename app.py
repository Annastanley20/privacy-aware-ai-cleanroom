import streamlit as st
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Set page configuration
st.set_page_config(page_title="Privacy-Aware AI Cleanroom", layout="wide")

# App Title
st.title("🔒 Privacy-Aware AI Cleanroom & Synthesizer")
st.markdown("""
Welcome to the **Privacy-Aware AI Cleanroom**. This intelligent system allows you to upload sensitive datasets, 
trains a privacy-preserving generative AI model, and produces high-fidelity **Synthetic Data** 
that protects user privacy while maintaining statistical utility.
""")

st.write("---")

# Sidebar Configuration for File Upload and Controls
st.sidebar.header("🛠️ Cleanroom Controls")
uploaded_file = st.sidebar.file_uploader("Upload your sensitive CSV dataset", type=["csv"])

# Target rows to generate
sample_rows = st.sidebar.slider("Number of Synthetic Rows to Generate", min_value=10, max_value=200, value=50, step=10)

# Load data based on user input
if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset uploaded successfully!")
else:
    # Fallback to local default data if no file is uploaded
    if pd.io.common.file_exists('original_data.csv'):
        df_original = pd.read_csv('original_data.csv')
        st.sidebar.info("💡 Using default sample dataset. Upload a CSV to test your own data.")
    else:
        st.error("⚠️ Sample dataset not found. Please upload a CSV file to begin.")
        st.stop()

# --- RUN THE AI ENGINE ---
@st.cache_data(show_spinner="🤖 AI is analyzing and synthesizing data... Please wait.")
def generate_synthetic_data(data, rows):
    # Detect Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    
    # Check if ID column exists to update its status
    id_cols = [col for col in data.columns if 'id' in col.lower() or 'key' in col.lower()]
    if id_cols:
        metadata.update_column(column_name=id_cols[0], sdtype='id')
        
    # Fit Synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    
    # Sample new safe records
    synthetic_df = synthesizer.sample(num_rows=rows)
    return synthetic_df

# Generate data using our cached function
df_synthetic = generate_synthetic_data(df_original, sample_rows)

# --- DISPLAY DATA TABLES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sensitive Input Dataset (Sample)")
    st.dataframe(df_original.head(10), use_container_width=True)
    st.caption(f"Total Original Records: {len(df_original)} rows")
    
with col2:
    st.subheader("🛡️ Privacy-Safe Synthetic Dataset (Generated)")
    st.dataframe(df_synthetic.head(10), use_container_width=True)
    st.caption(f"Total Synthetic Records: {len(df_synthetic)} rows | 100% Artificial")

st.write("---")

# --- VISUALIZATIONS SECTION ---
st.subheader("📈 Privacy vs. Utility Validation Graphs")
st.markdown("Compare how the AI preserved the mathematical shapes and trends without copying real identities.")

# Check if required columns exist for plotting
if 'Age' in df_original.columns and 'Monthly_Income' in df_original.columns:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("📊 **Age Distribution Trend**")
        # Prepare Age comparison data
        age_orig = df_original['Age'].value_counts().sort_index().rename('Original')
        age_synth = df_synthetic['Age'].value_counts().sort_index().rename('Synthetic')
        age_df = pd.concat([age_orig, age_synth], axis=1).fillna(0)
        st.line_chart(age_df)
        
    with chart_col2:
        st.write("💵 **Average Monthly Income Comparison**")
        # Prepare Income comparison metrics
        avg_income = pd.DataFrame({
            'Dataset': ['Original Data', 'Synthetic Data'],
            'Average Income': [df_original['Monthly_Income'].mean(), df_synthetic['Monthly_Income'].mean()]
        }).set_index('Dataset')
        st.bar_chart(avg_income)

else:
    st.warning("📊 Visualization plots require 'Age' and 'Monthly_Income' columns. Uploaded data differs.")

# --- DOWNLOAD BUTTON FOR SYNTHETIC DATA ---
st.write("---")
st.subheader("📥 Export Safe Data")
csv_data = df_synthetic.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Generated Synthetic CSV",
    data=csv_data,
    file_name="cleanroom_synthetic_data.csv",
    mime="text/csv"
)