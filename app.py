import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="DataSpark | AI-Powered AutoML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Mock Gemini API Call ---
def get_gemini_response(prompt):
    """Simulates a call to the Gemini API."""
    time.sleep(1) # Simulate network latency
    if "summary" in prompt.lower():
        return "This dataset appears robust with minimal data quality issues. The primary numerical features show moderate correlation, suggesting they will be valuable for predictive modeling. Key actions would involve standardizing numerical features before training."
    elif "rename" in prompt.lower():
        headers_str = prompt.split("Headers:")[1].strip()
        headers = [h.strip() for h in headers_str.split(',')]
        mock_rename = {h: h.replace("_", " ").title().replace(" ", "") for h in headers}
        return json.dumps(mock_rename)
    elif "interpretation" in prompt.lower():
        return "This model demonstrates strong performance, effectively capturing the patterns in the data. The most influential features identified are critical drivers for the target variable. For further improvement, consider hyperparameter tuning or creating interaction features from the top predictors."
    return "AI response could not be generated."

# --- Data Analysis Engine ---
def analyze_data(df):
    if not isinstance(df, pd.DataFrame) or df.empty: return {}
    n_rows, n_cols = df.shape
    column_details = {}
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        non_empty_count = n_rows - missing
        if pd.api.types.is_numeric_dtype(dtype): col_type = 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype) or (df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() / non_empty_count > 0.8): col_type = 'date'
        else: col_type = 'categorical'
        column_details[col] = {'missing': missing, 'missingPercentage': (missing / n_rows) * 100, 'dtype': col_type}
    numerical_cols = [col for col, details in column_details.items() if details['dtype'] == 'numerical']
    basic_stats = df[numerical_cols].describe().transpose().to_dict('index') if numerical_cols else {}
    correlation_matrix = df[numerical_cols].corr() if numerical_cols else pd.DataFrame()
    return {'shape': {'rows': n_rows, 'columns': n_cols}, 'column_details': column_details, 'basic_stats': basic_stats, 'correlation_matrix': correlation_matrix}

# --- Mock Model Training Engine ---
def train_models_mock(df, features, target, problem_type, selected_models):
    """Simulates training multiple models and returns realistic mock results."""
    time.sleep(2) # Simulate training time
    results = []
    
    if problem_type == 'Regression':
        base_r2 = 0.65 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_r2 + i * 0.02 + np.random.rand() * 0.05
            results.append({
                'name': model_name,
                'score': score,
                'metrics': {
                    'R-Squared': score,
                    'Adjusted R-Squared': score - 0.01,
                    'Mean Squared Error': np.random.uniform(100, 200) / (i + 1),
                    'Mean Absolute Error': np.random.uniform(10, 20) / (i + 1)
                },
                'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]
            })
    else: # Classification
        base_acc = 0.75 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_acc + i * 0.02 + np.random.rand() * 0.05
            results.append({
                'name': model_name,
                'score': score,
                'metrics': {
                    'Accuracy': score,
                    'Precision': score - np.random.uniform(0.02, 0.05),
                    'Recall': score + np.random.uniform(0.01, 0.03),
                    'F1-Score': score - np.random.uniform(0.01, 0.02)
                },
                'confusion_matrix': np.random.randint(0, 100, size=(2, 2)),
                'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]
            })
    return results

# --- Main App UI ---
st.title("⚡ DataSpark: AI-Powered AutoML")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.analysis = None
    st.session_state.file_name = None

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name
        with st.spinner('Analyzing data...'):
            st.session_state.analysis = analyze_data(df)
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.session_state.df = None

if st.session_state.df is not None:
    analysis = st.session_state.analysis
    df = st.session_state.df

    st.header(f"Analysis of: {st.session_state.file_name}")
    
    tabs = st.tabs(["Overview", "Preprocessing", "Feature Engineering", "Distributions", "Correlations", "🤖 Model Building"])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("AI-Generated Summary")
        with st.spinner("AI is generating insights..."):
            summary_prompt = f"As a senior data scientist, summarize a dataset with these characteristics: {json.dumps(analysis['shape'])}."
            st.markdown(get_gemini_response(summary_prompt))
        st.subheader("Dataset Preview (First 5 Rows)")
        st.dataframe(df.head())

    # --- Preprocessing Tab ---
    with tabs[1]:
        st.subheader("Data Cleaning & Preparation")
        st.info("Actions applied here will update the dataset for modeling.")
        if st.button("Apply All Recommended Preprocessing"):
            st.success("All preprocessing steps applied! Your dataset is now cleaner.")
        # Placeholder for more detailed preprocessing UI

    # --- Feature Engineering Tab ---
    with tabs[2]:
        st.subheader("Feature Engineering")
        st.info("Create new features to improve model performance.")
        # Placeholder

    # --- Distributions Tab ---
    with tabs[3]:
        st.subheader("Column Distributions")
        dist_col = st.selectbox("Select a column to visualize", df.columns, key="dist_select")
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
            fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough numerical columns to calculate correlations.")

    # --- Model Building Tab ---
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
                    target_variable = st.selectbox("Target Variable", [""] + list(df.columns))
                
                if target_variable:
                    with col2:
                        default_features = [col for col in df.columns if col != target_variable and df[col].nunique() > 1 and (df[col].nunique() / len(df) < 0.5)]
                        selected_features = st.multiselect("Features", [c for c in df.columns if c != target_variable], default=default_features)

                    st.subheader("4. Select Models to Train")
                    if is_regression:
                        model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
                    else:
                        model_options = ["Logistic Regression", "K-Nearest Neighbors", "SVM", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
                    selected_models = st.multiselect("Algorithms", model_options, default=model_options)

                    if st.button("Train Selected Models", type="primary", use_container_width=True):
                        with st.spinner("Training models... This may take a moment."):
                            st.session_state.model_results = train_models_mock(df, selected_features, target_variable, "Regression" if is_regression else "Classification", selected_models)

        else: # Unsupervised
            st.subheader("2. Unsupervised Learning Setup")
            unsupervised_models = st.multiselect("Select Models to Run", ["K-Means Clustering", "Principal Component Analysis (PCA)"])
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

                st.subheader("✨ AI Interpretation")
                with st.spinner("Generating AI interpretation..."):
                    interp_prompt = f"Interpret these model results: Model='{model['name']}', Score={model['score']:.3f}, Top Features={', '.join([f['feature'] for f in model['feature_importance']])}. Explain the performance and key drivers."
                    st.markdown(get_gemini_response(interp_prompt))


else:
    st.sidebar.info("Upload a CSV file to begin your analysis.")
    st.info("Welcome to DataSpark! Please upload a file to get started.")



