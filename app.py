import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="DataSpark | AI-Powered EDA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Gemini API Call Function ---
def get_gemini_response(prompt):
    """Calls the Gemini API to get a text response."""
    try:
        # NOTE: In a real Streamlit deployment, use st.secrets for the API key.
        # For this environment, we will use the built-in fetch.
        # This is a placeholder for the actual API call logic.
        # In a local Streamlit environment, you would use the google.generativeai library.
        # For this interactive canvas, we'll simulate the call.
        
        # This is a mock response for demonstration in this environment.
        # The real API call would be more complex.
        if "summary" in prompt.lower():
            return "This dataset appears to be in good condition. Key insights include..."
        elif "rename" in prompt.lower():
            # Extract headers from prompt for mock response
            headers_str = prompt.split("Headers:")[1].strip()
            headers = [h.strip() for h in headers_str.split(',')]
            mock_rename = {h: h.replace("_", " ").title().replace(" ", "") for h in headers}
            return json.dumps(mock_rename)
        return "AI response could not be generated."

    except Exception as e:
        st.error(f"Error calling AI model: {e}")
        return "AI response failed."


# --- Data Analysis Engine ---
def analyze_data(df):
    """Performs a comprehensive analysis of the dataframe."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    n_rows, n_cols = df.shape
    
    # Column Details
    column_details = {}
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        non_empty_count = n_rows - missing
        
        # Type detection
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype) or \
             (df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() / non_empty_count > 0.8):
            col_type = 'date'
        else:
            col_type = 'categorical'
            
        column_details[col] = {
            'missing': missing,
            'missingPercentage': (missing / n_rows) * 100,
            'dtype': col_type
        }

    # Basic Stats & Outliers
    numerical_cols = [col for col, details in column_details.items() if details['dtype'] == 'numerical']
    basic_stats = df[numerical_cols].describe().transpose().to_dict('index')
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers[col] = {'count': outlier_count, 'percentage': (outlier_count / n_rows) * 100}

    # Value Counts
    categorical_cols = [col for col, details in column_details.items() if details['dtype'] == 'categorical']
    value_counts = {col: df[col].value_counts().head(10).to_dict() for col in categorical_cols}

    # Correlation Matrix
    correlation_matrix = df[numerical_cols].corr()

    return {
        'shape': {'rows': n_rows, 'columns': n_cols},
        'missing_summary': {'total_missing': df.isnull().sum().sum()},
        'column_details': column_details,
        'basic_stats': basic_stats,
        'value_counts': value_counts,
        'outliers': outliers,
        'duplicate_rows': df.duplicated().sum(),
        'correlation_matrix': correlation_matrix
    }

# --- Main App UI ---
st.title("⚡ DataSpark: AI-Powered EDA")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.analysis = None
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name
        with st.spinner('Analyzing data...'):
            st.session_state.analysis = analyze_data(df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.session_state.df = None

if st.session_state.df is not None:
    st.success(f"Successfully loaded and analyzed **{st.session_state.file_name}**")
    
    analysis = st.session_state.analysis
    df = st.session_state.df

    # --- Dashboard Tabs ---
    tabs = st.tabs(["Overview", "Preprocessing", "Feature Engineering", "Distributions", "Correlations", "Model Building"])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("AI-Generated Summary")
        with st.spinner("AI is generating insights..."):
            summary_prompt = f"As a senior data scientist, provide a 1-2 paragraph summary of a dataset with these characteristics: {json.dumps(analysis['shape'])}. Be concise and actionable."
            ai_summary = get_gemini_response(summary_prompt)
            st.markdown(ai_summary)

        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{analysis['shape']['rows']:,}")
        col2.metric("Columns", f"{analysis['shape']['columns']:,}")
        col3.metric("Missing Cells", f"{analysis['missing_summary']['total_missing']:,}")
        col4.metric("Duplicate Rows", f"{analysis['duplicate_rows']:,}")

        st.subheader("Dataset Preview (First 5 Rows)")
        st.dataframe(df.head())

    # --- Preprocessing Tab ---
    with tabs[1]:
        st.subheader("Intelligent Preprocessing")
        st.info("Apply cleaning actions and the analysis will update automatically.")

        # Column Renaming
        with st.expander("✨ AI-Powered Column Renaming"):
            if st.button("Suggest Better Names"):
                with st.spinner("AI is thinking of better names..."):
                    rename_prompt = f"Given the following CSV column headers, suggest clearer, more conventional names. Return only a JSON object mapping the old name to the new name. Headers: {', '.join(df.columns)}"
                    st.session_state.suggested_names = json.loads(get_gemini_response(rename_prompt))
            
            if 'suggested_names' in st.session_state and st.session_state.suggested_names:
                st.write("AI Suggestions:")
                st.json(st.session_state.suggested_names)
                if st.button("Apply Renames"):
                    df.rename(columns=st.session_state.suggested_names, inplace=True)
                    st.session_state.df = df
                    st.session_state.analysis = analyze_data(df)
                    del st.session_state.suggested_names
                    st.success("Columns renamed!")
                    st.experimental_rerun()

        # Missing Value Imputation
        with st.expander("Handle Missing Values"):
            missing_cols = {col: details for col, details in analysis['column_details'].items() if details['missing'] > 0}
            if not missing_cols:
                st.write("No missing values found. Great!")
            else:
                for col, details in missing_cols.items():
                    col1, col2 = st.columns([1, 2])
                    col1.write(f"**{col}** ({details['missing']:,} missing)")
                    if details['dtype'] == 'numerical':
                        method = col2.selectbox(f"Impute {col}", ["Mean", "Median", "Zero"], key=f"impute_{col}")
                        if st.button(f"Apply to {col}", key=f"apply_{col}"):
                            if method == "Mean":
                                fill_value = df[col].mean()
                            elif method == "Median":
                                fill_value = df[col].median()
                            else:
                                fill_value = 0
                            df[col].fillna(fill_value, inplace=True)
                            st.session_state.df = df
                            st.session_state.analysis = analyze_data(df)
                            st.experimental_rerun()

    # --- Feature Engineering Tab ---
    with tabs[2]:
        st.subheader("Feature Engineering Suggestions")
        date_cols = [col for col, details in analysis['column_details'].items() if details['dtype'] == 'date']
        if not date_cols:
            st.write("No date columns detected for feature extraction.")
        else:
            for col in date_cols:
                if st.button(f"Extract Parts from '{col}'"):
                    dt_col = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = dt_col.dt.year
                    df[f'{col}_month'] = dt_col.dt.month
                    df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                    st.session_state.df = df
                    st.session_state.analysis = analyze_data(df)
                    st.success(f"Created new features from {col}")
                    st.experimental_rerun()

    # --- Distributions Tab ---
    with tabs[3]:
        st.subheader("Column Distributions")
        dist_col = st.selectbox("Select a column to visualize", df.columns)
        if analysis['column_details'][dist_col]['dtype'] == 'numerical':
            fig = px.histogram(df, x=dist_col, title=f"Distribution of {dist_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            vc = df[dist_col].value_counts().head(20)
            fig = px.bar(vc, x=vc.index, y=vc.values, labels={'x': dist_col, 'y': 'Count'}, title=f"Value Counts for {dist_col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Correlations Tab ---
    with tabs[4]:
        st.subheader("Correlation Matrix")
        corr_matrix = analysis['correlation_matrix']
        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig.update_layout(title="Correlation Between Numerical Features")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numerical columns to calculate correlations.")
            
    # --- Model Building Tab ---
    with tabs[5]:
        st.subheader("🤖 AutoML Workspace")
        target_variable = st.selectbox("1. Select Target Variable (what to predict)", [""] + list(df.columns))

        if target_variable:
            problem_type = analysis['column_details'][target_variable]['dtype']
            st.success(f"Problem Type Detected: **{problem_type.capitalize()}**")

            # Feature Selection
            st.subheader("2. Select Features for Training")
            default_features = [col for col in df.columns if col != target_variable and df[col].nunique() > 1 and df[col].nunique() < 50]
            selected_features = st.multiselect("Features", df.columns, default=default_features)

            if st.button("Train Models", type="primary"):
                with st.spinner("Training models... This may take a moment."):
                    # Mock training results for demonstration
                    st.session_state.model_results = {
                        'models': [
                            {'name': 'Logistic/Linear Regression', 'score': np.random.rand(), 'metrics': {'MSE': 12.3, 'MAE': 2.5}},
                            {'name': 'Decision Tree', 'score': np.random.rand(), 'metrics': {'MSE': 10.1, 'MAE': 2.1}},
                            {'name': 'Random Forest', 'score': np.random.rand(), 'metrics': {'MSE': 8.7, 'MAE': 1.9}},
                        ],
                        'problem_type': problem_type,
                        'best_model': 'Random Forest'
                    }
        
        if 'model_results' in st.session_state and st.session_state.model_results:
            st.subheader("3. Model Results")
            results = st.session_state.model_results
            metric_name = "R-Squared" if results['problem_type'] == 'numerical' else "Accuracy"
            
            for model in sorted(results['models'], key=lambda x: x['score'], reverse=True):
                is_best = model['name'] == results['best_model']
                with st.expander(f"{'🏆 ' if is_best else ''}{model['name']} ({metric_name}: {model['score']:.2f})", expanded=is_best):
                    st.write(f"**Performance Metrics for {model['name']}**")
                    st.json(model['metrics'])
                    st.write("**AI Interpretation**")
                    st.info("AI analysis of the model's performance, feature importance, and conclusion would appear here.")

else:
    st.info("Upload a CSV file to get started with your analysis.")

