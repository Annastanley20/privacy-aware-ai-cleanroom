import streamlit as st
import pandas as pd
import os

# Set page configuration for a wide dashboard layout
st.set_page_config(page_title="Privacy-Aware AI Cleanroom", layout="wide")

# App Title and Description
st.title("🔒 Privacy-Aware AI Cleanroom & Synthesizer")
st.markdown("""
Welcome to the **Privacy-Aware AI Cleanroom**. This intelligent system reads sensitive production data, 
analyzes its underlying statistical distributions, and generates high-fidelity **Synthetic Data** 
that protects individual privacy while preserving mathematical utility.
""")

st.write("---")

# Check if data files exist from our training session
if os.path.exists('original_data.csv') and os.path.exists('synthetic_data.csv'):
    
    # Load the datasets
    df_original = pd.read_csv('original_data.csv')
    df_synthetic = pd.read_csv('synthetic_data.csv')
    
    # Create layout columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Original Sensitive Dataset (Sample)")
        st.dataframe(df_original.head(10), use_container_width=True)
        st.caption(f"Total Records: {len(df_original)} rows | Contains PII (Personally Identifiable Information)")
        
    with col2:
        st.subheader("🛡️ Generated Safe Synthetic Dataset (Sample)")
        st.dataframe(df_synthetic.head(10), use_container_width=True)
        st.caption(f"Total Records: {len(df_synthetic)} rows | 100% Privacy-Safe (No real individuals)")

    st.write("---")
    
    # Quick Statistical Check
    st.subheader("📈 Statistical Mean Comparison")
    
    metrics_data = {
        'Metric': ['Average Age', 'Average Monthly Income', 'Average Loan Amount'],
        'Original Data': [round(df_original['Age'].mean(), 1), f"{round(df_original['Monthly_Income'].mean(), 2):,}", f"{round(df_original['Loan_Amount'].mean(), 2):,}"],
        'Synthetic Data': [round(df_synthetic['Age'].mean(), 1), f"{round(df_synthetic['Monthly_Income'].mean(), 2):,}", f"{round(df_synthetic['Loan_Amount'].mean(), 2):,}"]
    }
    st.table(pd.DataFrame(metrics_data))

else:
    st.error("⚠️ Datasets not found! Please run the Jupyter Notebook (`synthesizer_test.ipynb`) first to generate the CSV files.")