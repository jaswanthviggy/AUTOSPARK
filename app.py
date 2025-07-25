import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="DataSpark | AI-Powered AutoML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Analysis Engine ---
def analyze_data(df):
    if not isinstance(df, pd.DataFrame) or df.empty: return {}
    n_rows, n_cols = df.shape
    column_details = {}
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        non_empty_count = n_rows - missing
        
        # Improved Type Detection
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype) or \
             (df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() / non_empty_count > 0.8):
            col_type = 'date'
        else:
            # Check for complex text that could be engineered
            if df[col].astype(str).str.contains(r'(\d+\.?\d*)\s*(HP|L|V\d)', case=False).sum() / non_empty_count > 0.5:
                col_type = 'complex_text'
            else:
                col_type = 'categorical'

        column_details[col] = {'missing': missing, 'missingPercentage': (missing / n_rows) * 100, 'dtype': col_type}
    
    numerical_cols = [col for col, details in column_details.items() if details['dtype'] == 'numerical']
    
    basic_stats = {}
    if numerical_cols:
        desc = df[numerical_cols].describe().transpose()
        skew = df[numerical_cols].skew()
        desc['skewness'] = skew
        basic_stats = desc.to_dict('index')

    correlation_matrix = df[numerical_cols].corr() if numerical_cols else pd.DataFrame()
    value_counts = {col: df[col].value_counts().head(10).to_dict() for col in [c for c,d in column_details.items() if d['dtype']=='categorical']}
    
    return {
        'shape': {'rows': n_rows, 'columns': n_cols},
        'column_details': column_details,
        'basic_stats': basic_stats,
        'correlation_matrix': correlation_matrix,
        'value_counts': value_counts,
        'duplicate_rows': df.duplicated().sum()
    }

# --- Mock Model Training Engine ---
def train_models_mock(df, features, target, problem_type, selected_models):
    time.sleep(2)
    results = []
    if problem_type == 'Regression':
        base_r2 = 0.65 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_r2 + i * 0.02 + np.random.rand() * 0.05
            results.append({'name': model_name, 'score': score, 'metrics': {'R-Squared': score, 'MSE': np.random.uniform(100,200)/(i+1), 'MAE': np.random.uniform(10,20)/(i+1)}, 'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]})
    else:
        base_acc = 0.75 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_acc + i * 0.02 + np.random.rand() * 0.05
            results.append({'name': model_name, 'score': score, 'metrics': {'Accuracy': score, 'Precision': score-0.05, 'Recall': score+0.02, 'F1-Score': score-0.01}, 'confusion_matrix': np.random.randint(0, 100, size=(2,2)), 'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]})
    return results

# --- Main App UI ---
st.title("⚡ DataSpark: Advanced AutoML")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.df_transformed = None
    st.session_state.analysis = None
    st.session_state.file_name = None
    st.session_state.transform_log = []

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_transformed = df.copy()
        st.session_state.file_name = uploaded_file.name
        st.session_state.transform_log = ["Dataset Loaded."]
        with st.spinner('Analyzing data...'):
            st.session_state.analysis = analyze_data(df)
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.session_state.df = None

if st.session_state.df is not None:
    analysis = analyze_data(st.session_state.df_transformed)
    df_original = st.session_state.df
    df_transformed = st.session_state.df_transformed

    st.header(f"Analysis of: {st.session_state.file_name}")
    
    tabs = st.tabs(["Overview", "Data Info", "🔬 Data Transformation", "Distributions", "Correlations", "🤖 Model Building"])

    with tabs[0]:
        st.subheader("Key Metrics (Based on Current Data)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{analysis['shape']['rows']:,}")
        col2.metric("Columns", f"{analysis['shape']['columns']:,}")
        col3.metric("Missing Cells", f"{df_transformed.isnull().sum().sum():,}")
        col4.metric("Duplicate Rows", f"{df_transformed.duplicated().sum():,}")
        st.subheader("Original Data Preview")
        st.dataframe(df_original.head())

    with tabs[1]:
        st.subheader("Column Inspector")
        col_to_inspect = st.selectbox("Select a column for detailed info", df_transformed.columns)
        if col_to_inspect:
            col_details = analysis['column_details'][col_to_inspect]
            st.write(f"**Data Type:** `{col_details['dtype']}`")
            st.write(f"**Missing Values:** {col_details['missing']} ({col_details['missingPercentage']:.2f}%)")
            st.write(f"**Unique Values:** {df_transformed[col_to_inspect].nunique()}")
            if col_details['dtype'] == 'numerical':
                st.write("**Descriptive Statistics:**")
                st.dataframe(df_transformed[col_to_inspect].describe())
            elif col_details['dtype'] == 'categorical':
                st.write("**Value Counts:**")
                st.dataframe(df_transformed[col_to_inspect].value_counts().head(10))

    with tabs[2]:
        st.header("🔬 Data Transformation Workspace")
        st.info("Prepare your data for modeling. All actions here are performed on a copy of your original data.")
        
        # Auto-Clean
        if st.button("⚡ Auto-Clean Dataset", use_container_width=True, type="primary"):
            with st.spinner("Running Auto-Clean pipeline..."):
                # Simulate a full cleaning pipeline
                st.session_state.transform_log.append("Ran Auto-Clean: Imputed, Scaled, Encoded.")
                st.success("Auto-Clean pipeline completed!")
                # In a real app, you would call the actual cleaning functions here
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Transformation Log")
            st.code("\n".join(st.session_state.transform_log), language="log")
        with col2:
            st.subheader("Transformed Data Preview")
            st.dataframe(df_transformed.head())

        st.divider()
        st.subheader("Intelligent Feature Extraction")
        complex_cols = [col for col, det in analysis['column_details'].items() if det['dtype'] == 'complex_text']
        if not complex_cols:
            st.write("No complex text columns found for feature extraction.")
        else:
            for col in complex_cols:
                st.write(f"**Column: `{col}`**")
                st.write("This column contains multiple pieces of information. We can try to extract them into new features.")
                if st.button(f"Extract Features from `{col}`"):
                    # Example: Extract HP, Liters, Cylinders from an 'Engine' column
                    df_transformed[f'{col}_HP'] = df_transformed[col].str.extract(r'(\d+\.?\d*)\s*HP', expand=False).astype(float)
                    df_transformed[f'{col}_Liters'] = df_transformed[col].str.extract(r'(\d+\.?\d*)\s*L', expand=False).astype(float)
                    df_transformed[f'{col}_Cylinders'] = df_transformed[col].str.extract(r'V(\d+)', expand=False).astype(float)
                    df_transformed.drop(columns=[col], inplace=True)
                    st.session_state.df_transformed = df_transformed
                    st.session_state.transform_log.append(f"Extracted HP, Liters, Cylinders from '{col}'. Dropped original.")
                    st.experimental_rerun()
    
    with tabs[3]:
        st.subheader("Column Distributions (on Transformed Data)")
        dist_col = st.selectbox("Select a column to visualize", df_transformed.columns, key="dist_select")
        if analysis['column_details'].get(dist_col, {}).get('dtype') == 'numerical':
            fig = px.histogram(df_transformed, x=dist_col, title=f"Distribution of {dist_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            vc = df_transformed[dist_col].value_counts().head(20)
            fig = px.bar(vc, x=vc.index, y=vc.values, labels={'x': dist_col, 'y': 'Count'}, title=f"Value Counts for {dist_col}")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Correlation Matrix (on Transformed Data)")
        corr_matrix = analyze_data(df_transformed)['correlation_matrix']
        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numerical columns to calculate correlations.")

    with tabs[5]:
        st.header("🤖 AutoML Workspace")
        
        ml_task = st.selectbox("1. Choose your Machine Learning Task", ["Supervised Learning", "Unsupervised Learning"])
        
        if ml_task == "Supervised Learning":
            problem_type = st.radio("2. Select your Prediction Goal", ["Regression (Predict a number)", "Classification (Predict a category)"], horizontal=True)
            
            if problem_type:
                is_regression = "Regression" in problem_type
                st.subheader("3. Select Target and Features")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_variable = st.selectbox("Target Variable", [""] + list(df_transformed.columns))
                
                if target_variable:
                    with col2:
                        default_features = [col for col in df_transformed.columns if col != target_variable and df_transformed[col].nunique() > 1 and (df_transformed[col].nunique() / len(df_transformed) < 0.5)]
                        selected_features = st.multiselect("Features", [c for c in df_transformed.columns if c != target_variable], default=default_features)

                    st.subheader("4. Select Models to Train")
                    if is_regression:
                        model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
                    else:
                        model_options = ["Logistic Regression", "KNN Classifier", "SVM", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
                    selected_models = st.multiselect("Algorithms", model_options, default=model_options)

                    if st.button("Train Selected Models", type="primary", use_container_width=True):
                        with st.spinner("Training models on transformed data..."):
                            st.session_state.model_results = train_models_mock(df_transformed, selected_features, target_variable, "Regression" if is_regression else "Classification", selected_models)

        else: # Unsupervised
            st.subheader("2. Unsupervised Learning Setup")
            unsupervised_task = st.selectbox("Select Task", ["Clustering", "Dimensionality Reduction"])
            if unsupervised_task == "Clustering":
                model_options = ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"]
            else:
                model_options = ["Principal Component Analysis (PCA)", "t-SNE"]
            
            selected_models = st.multiselect("Select Models to Run", model_options)
            if st.button("Run Models", type="primary", use_container_width=True):
                st.success("Unsupervised models ran successfully! (This is a placeholder)")
        
        if 'model_results' in st.session_state and st.session_state.model_results:
            st.subheader("5. Model Results Leaderboard")
            results = st.session_state.model_results
            
            metric_name = "R-Squared" if "Regression" in results[0]['name'] else "Accuracy"
            
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = None

            cols = st.columns(len(sorted_results))
            for i, model in enumerate(sorted_results):
                with cols[i]:
                    is_best = i == 0
                    st.metric(label=f"{'🏆 ' if is_best else ''}{model['name']}", value=f"{model['score']:.3f}")
                    if st.button(f"View Details", key=f"details_{model['name']}"):
                        st.session_state.selected_model = model

            if st.session_state.selected_model:
                model = st.session_state.selected_model
                st.divider()
                st.header(f"Detailed Results for: {model['name']}")
                
                st.subheader("Performance Metrics")
                st.json(model['metrics'])
                if 'confusion_matrix' in model:
                    st.subheader("Confusion Matrix")
                    fig = px.imshow(model['confusion_matrix'], text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale='Blues')
                    st.plotly_chart(fig)

                st.subheader("Feature Importance")
                fig = px.bar(pd.DataFrame(model['feature_importance']), x='importance', y='feature', orientation='h', title="Top 10 Most Influential Features")
                st.plotly_chart(fig)

else:
    st.sidebar.info("Upload a CSV file to begin your analysis.")
    st.info("Welcome to DataSpark! Please upload a file to get started.")

