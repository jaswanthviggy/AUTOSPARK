import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Page Configuration ---
st.set_page_config(
    page_title="DataSpark | AI-Powered AutoML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MOCK GEMINI API ---
def get_gemini_response(prompt):
    """Simulates a call to the Gemini API for demonstration."""
    time.sleep(1.5) # Simulate network latency
    if "summary" in prompt.lower():
        return """
        This dataset appears to be of moderate quality with some immediate areas for improvement. 
        The presence of missing values and potential outliers in key numerical columns suggests that a robust preprocessing pipeline will be crucial for accurate modeling. 
        The categorical features have a reasonable number of unique values, making them good candidates for one-hot encoding. 
        **Next Steps:** Navigate to the 'Data Transformation' tab to clean and prepare the data. The 'Auto-Clean' feature is recommended as a strong starting point.
        """
    elif "interpretation" in prompt.lower():
        return """
        **Model Performance:**
        The model demonstrates strong predictive power, achieving a high score on the hold-out test set. This indicates that the selected features have a significant relationship with the target variable. The model appears to generalize well, suggesting it is not overfitting to the training data.

        **Key Drivers:**
        The feature importance chart clearly shows that `Feature_A` and `Feature_B` are the most influential predictors. Changes in these variables have the largest impact on the model's output. This insight is critical and suggests that any business strategy should focus heavily on these factors.

        **Conclusion & Recommendations:**
        This model is a reliable baseline for this predictive task. To further improve performance, consider experimenting with hyperparameter tuning on this model or creating interaction features from the top predictors (e.g., `Feature_A * Feature_B`). For deployment, this model is a strong candidate due to its balance of performance and interpretability.
        """
    return "AI response could not be generated."

# --- DATA ANALYSIS ENGINE ---
def analyze_data(df):
    if not isinstance(df, pd.DataFrame) or df.empty: return {}
    n_rows, n_cols = df.shape
    column_details = {}
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        non_empty_count = n_rows - missing
        
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype) or (df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}', na=False).sum() / non_empty_count > 0.8):
            col_type = 'date'
        else:
            # More specific regex for multi-feature extraction
            if df[col].astype(str).str.contains(r'(\d+\.?\d*)\s*HP.*(\d+\.?\d*)\s*L.*V(\d+)', case=False, na=False).sum() / non_empty_count > 0.5:
                col_type = 'multi_feature_text'
            elif df[col].astype(str).str.contains(r'\d', na=False).sum() / non_empty_count > 0.7:
                col_type = 'dirty_numerical'
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
    
    return {'shape': {'rows': n_rows, 'columns': n_cols}, 'column_details': column_details, 'basic_stats': basic_stats, 'correlation_matrix': correlation_matrix, 'value_counts': value_counts, 'duplicate_rows': df.duplicated().sum()}

# --- MOCK MODEL TRAINING ENGINE ---
def train_models_mock(df, features, target, problem_type, selected_models, test_size, random_state):
    st.session_state.transform_log.append(f"Starting training with test size {test_size} and random state {random_state}.")
    time.sleep(2)
    results = []
    if problem_type == 'Regression':
        base_r2 = 0.65 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_r2 + i * 0.02 + np.random.rand() * 0.05
            results.append({'name': model_name, 'score': score, 'metrics': {'R-Squared': score, 'Adjusted R-Squared': score - 0.01, 'MSE': np.random.uniform(100,200)/(i+1), 'MAE': np.random.uniform(10,20)/(i+1)}, 'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]})
    else:
        base_acc = 0.75 + np.random.rand() * 0.1
        for i, model_name in enumerate(selected_models):
            score = base_acc + i * 0.02 + np.random.rand() * 0.05
            results.append({'name': model_name, 'score': score, 'metrics': {'Accuracy': score, 'Precision': score-0.05, 'Recall': score+0.02, 'F1-Score': score-0.01}, 'confusion_matrix': np.random.randint(0, 100, size=(2,2)), 'feature_importance': sorted([{'feature': f, 'importance': np.random.rand()} for f in features], key=lambda x: x['importance'], reverse=True)[:10]})
    return results

# --- MAIN APP UI ---
st.title("⚡ DataSpark: Advanced AutoML")

# --- SESSION STATE INITIALIZATION ---
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.df_transformed = None
    st.session_state.analysis = None
    st.session_state.file_name = None
    st.session_state.transform_log = []
if 'nn_layers' not in st.session_state:
    st.session_state.nn_layers = [{'neurons': 64, 'activation': 'ReLU'}]

# --- FILE UPLOADER ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        try:
            df = pd.read_csv(uploaded_file, thousands=',')
            st.session_state.df = df
            st.session_state.df_transformed = df.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.transform_log = ["Dataset Loaded."]
            with st.spinner('Analyzing data...'):
                st.session_state.analysis = analyze_data(df)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.session_state.df = None

# --- MAIN DASHBOARD ---
if st.session_state.df is not None:
    analysis = analyze_data(st.session_state.df_transformed)
    df_original = st.session_state.df
    df_transformed = st.session_state.df_transformed

    st.header(f"Analysis of: {st.session_state.file_name}")
    
    tabs = st.tabs(["Overview", "Data Info", "🔬 Data Transformation", "Distributions", "Correlations", "🤖 Model Building", "🧠 Deep Learning", "🚀 Export & Deploy"])

    with tabs[0]:
        st.subheader("Data Health Score")
        missing_pct = (df_transformed.isnull().sum().sum() / (df_transformed.shape[0] * df_transformed.shape[1])) * 100
        health_score = max(0, 100 - missing_pct * 5 - (analysis['duplicate_rows'] / analysis['shape']['rows']) * 100)
        st.progress(int(health_score))
        st.metric("Health Score", f"{health_score:.1f}%")

        st.subheader("AI-Generated Summary")
        with st.spinner("AI is generating insights..."):
            summary_prompt = f"As a senior data scientist, summarize a dataset with these characteristics: {json.dumps(analysis['shape'])}."
            st.markdown(get_gemini_response(summary_prompt))
        
        st.subheader("Key Metrics")
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
        if col_to_inspect and col_to_inspect in analysis['column_details']:
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
        st.info("Prepare your data for modeling. Actions here modify a copy of your original data.")
        
        if st.button("⚡ Auto-Clean Dataset", use_container_width=True, type="primary"):
            with st.spinner("Running Auto-Clean pipeline..."):
                temp_df = st.session_state.df_transformed.copy()
                log = st.session_state.transform_log
                
                num_cols = [c for c, d in analysis['column_details'].items() if d['dtype'] == 'numerical' and d['missing'] > 0]
                cat_cols = [c for c, d in analysis['column_details'].items() if d['dtype'] == 'categorical' and d['missing'] > 0]
                
                for col in num_cols:
                    if abs(analysis['basic_stats'][col]['skewness']) > 1:
                        fill_value = temp_df[col].median()
                        log.append(f"Auto: Imputed '{col}' with median ({fill_value:.2f}) due to skew.")
                    else:
                        fill_value = temp_df[col].mean()
                        log.append(f"Auto: Imputed '{col}' with mean ({fill_value:.2f}).")
                    temp_df[col].fillna(fill_value, inplace=True)

                if cat_cols:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    temp_df[cat_cols] = pd.DataFrame(imputer_cat.fit_transform(temp_df[cat_cols]), columns=cat_cols, index=temp_df.index)
                    log.append("Auto: Imputed missing categorical data with mode.")
                
                numerical_cols_to_scale = [c for c, d in analysis['column_details'].items() if d['dtype'] == 'numerical']
                if numerical_cols_to_scale:
                    scaler = StandardScaler()
                    temp_df[numerical_cols_to_scale] = scaler.fit_transform(temp_df[numerical_cols_to_scale])
                    log.append("Auto: Scaled numerical features with StandardScaler.")

                st.session_state.df_transformed = temp_df
                st.session_state.transform_log = log
                st.success("Auto-Clean pipeline completed!")
                st.rerun()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Transformation Log")
            st.code("\n".join(st.session_state.transform_log), language="log")
        with col2:
            st.subheader("Transformed Data Preview")
            st.dataframe(df_transformed.head())

        st.divider()
        st.subheader("Intelligent Feature Extraction & Cleaning")
        complex_cols = [col for col, det in analysis['column_details'].items() if det['dtype'] == 'multi_feature_text' or det['dtype'] == 'dirty_numerical']
        if not complex_cols:
            st.write("No complex text columns found for feature extraction.")
        else:
            for col in complex_cols:
                with st.container():
                    st.write(f"**Column: `{col}`**")
                    st.write("This column contains numbers mixed with text. We can attempt to clean it.")
                    if st.button(f"Clean & Convert '{col}' to Number", key=f"clean_{col}"):
                        df_copy = st.session_state.df_transformed.copy()
                        df_copy[col] = df_copy[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                        st.session_state.df_transformed = df_copy
                        st.session_state.transform_log.append(f"Cleaned and converted '{col}' to a numerical type.")
                        st.rerun()
    
    with tabs[3]:
        st.subheader("Column Distributions (on Transformed Data)")
        if not df_transformed.empty:
            dist_col = st.selectbox("Select a column to visualize", df_transformed.columns, key="dist_select")
            if dist_col and analysis['column_details'].get(dist_col, {}).get('dtype') == 'numerical':
                fig = px.histogram(df_transformed, x=dist_col, title=f"Distribution of {dist_col}")
                st.plotly_chart(fig, use_container_width=True)
            elif dist_col:
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

                    st.subheader("4. Configure Training Parameters")
                    with st.expander("Set Training Options"):
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                        random_state = st.number_input("Random State (for reproducibility)", value=42)
                        st.checkbox("Enable Hyperparameter Tuning (Coming Soon)", disabled=True)

                    st.subheader("5. Select Models to Train")
                    if is_regression:
                        model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
                    else:
                        model_options = ["Logistic Regression", "KNN Classifier", "SVM", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
                    selected_models = st.multiselect("Algorithms", model_options, default=model_options)

                    if st.button("Train Selected Models", type="primary", use_container_width=True):
                        with st.spinner("Training models on transformed data..."):
                            st.session_state.model_results = train_models_mock(df_transformed, selected_features, target_variable, "Regression" if is_regression else "Classification", selected_models, test_size, random_state)

        else: # Unsupervised
            st.subheader("2. Unsupervised Learning Setup")
            unsupervised_task = st.selectbox("Select Task", ["Clustering", "Dimensionality Reduction"])
            if unsupervised_task == "Clustering": model_options = ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"]
            else: model_options = ["Principal Component Analysis (PCA)", "t-SNE"]
            
            selected_models = st.multiselect("Select Models to Run", model_options)
            if st.button("Run Models", type="primary", use_container_width=True):
                st.success("Unsupervised models ran successfully! (This is a placeholder)")
        
        if 'model_results' in st.session_state and st.session_state.model_results:
            st.subheader("6. Model Results Leaderboard")
            results = st.session_state.model_results
            
            is_regression_results = 'R-Squared' in results[0]['metrics']
            metric_name = "R-Squared" if is_regression_results else "Accuracy"
            
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

            if 'selected_model' not in st.session_state: st.session_state.selected_model = None

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

    with tabs[6]:
        st.header("🧠 Deep Learning Workspace")
        st.info("This is an advanced workspace for building and training Artificial Neural Networks (ANNs) for your tabular data.")
        
        st.subheader("Pro-Tip: When to use Deep Learning")
        st.markdown("""
        - **Good for:** Complex, non-linear relationships in large datasets.
        - **Not for:** Small datasets (prone to overfitting) or when model interpretability is the top priority.
        - **CNNs/RNNs:** These are specialized for image/sequence data. For this tabular dataset, a standard ANN (or MLP) is the best choice.
        """)

        st.subheader("1. Build Your Network Architecture")
        
        for i, layer in enumerate(st.session_state.nn_layers):
            cols = st.columns([3, 2, 1])
            layer['neurons'] = cols[0].slider(f"Neurons in Hidden Layer {i+1}", 8, 512, layer['neurons'], 8, key=f"neurons_{i}")
            layer['activation'] = cols[1].selectbox(f"Activation Function {i+1}", ["ReLU", "Tanh", "Sigmoid"], key=f"activation_{i}")
            if len(st.session_state.nn_layers) > 1:
                if cols[2].button("Remove", key=f"remove_layer_{i}"):
                    st.session_state.nn_layers.pop(i)
                    st.rerun()

        if st.button("Add Hidden Layer"):
            st.session_state.nn_layers.append({'neurons': 32, 'activation': 'ReLU'})
            st.rerun()

        st.subheader("2. Train the Network")
        if st.button("Train ANN Model", type="primary", use_container_width=True):
            with st.spinner("Training Neural Network... (This is a simulation)"):
                # Simulate training and create a loss chart
                epochs = 50
                loss_data = pd.DataFrame({
                    'Epoch': range(1, epochs + 1),
                    'Training Loss': np.geomspace(0.8, 0.1, epochs) + np.random.rand(epochs) * 0.1,
                    'Validation Loss': np.geomspace(0.7, 0.15, epochs) + np.random.rand(epochs) * 0.1,
                })
                st.session_state.training_chart = loss_data
                st.session_state.shap_plot = True # Simulate that a SHAP plot is ready

        if 'training_chart' in st.session_state:
            st.subheader("Live Training Progress (Simulated)")
            st.line_chart(st.session_state.training_chart, x='Epoch')
        
        if 'shap_plot' in st.session_state:
            st.subheader("Model Interpretation (SHAP Summary Plot)")
            st.info("A SHAP plot shows the impact of each feature on the model's predictions. Red dots represent high feature values, blue dots represent low ones.")
            # Display a placeholder image for the SHAP plot
            st.image("https://i.imgur.com/kIBi9sN.png", caption="Example SHAP Summary Plot")

    with tabs[7]:
        st.header("🚀 Export & Deploy")
        st.subheader("Download Artifacts")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=df_transformed.to_csv(index=False).encode('utf-8'),
                file_name=f"cleaned_{st.session_state.file_name}",
                mime='text/csv',
                use_container_width=True
            )
        with col2:
            st.button("Download Model Report (PDF)", disabled=True, use_container_width=True)

        st.subheader("Get Prediction Code")
        st.info("Use the following Python code snippet to make predictions with your best model.")
        st.code("""
# This is a placeholder for prediction code.
# In a real app, this would be populated with a scikit-learn pipeline.

def predict(new_data):
    # model.predict(new_data)
    print("Prediction logic would be here.")

        """, language="python")

else:
    st.sidebar.info("Upload a CSV file to begin your analysis.")
    st.info("Welcome to DataSpark! Please upload a file to get started.")

