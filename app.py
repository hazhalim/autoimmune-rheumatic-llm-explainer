# Installs
# pip install streamlit joblib sdv pandas numpy torch

# Imports
# Base disease diagnosis imports
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import traceback
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# SHAP and LIME imports
# import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import lime
import dill

# Gemini imports
from google import genai
from PIL import Image
from google.genai.types import GenerateContentConfig
from io import BytesIO
import os
import kaleido
from sklearn.neighbors import KNeighborsClassifier

# Unused imports
# from sdv.utils import load_synthesizer
# import torch
# from sdv.sampling import Condition

# Constants
AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES = [
    'Ankylosing Spondylitis',      # 0
    'Normal',                      # 1
    'Psoriatic Arthritis',         # 2
    'Reactive Arthritis',          # 3
    'Rheumatoid Arthritis',        # 4
    "Sjögren's Syndrome",          # 5
    'Systemic Lupus Erythematosus' # 6
]

AUTOIMMUNE_RHEUMATIC_DISEASE_MAPPING = {
    0: 'Ankylosing Spondylitis',
    1: 'Normal',
    2: 'Psoriatic Arthritis',
    3: 'Reactive Arthritis',
    4: 'Rheumatoid Arthritis',
    5: "Sjögren's Syndrome",
    6: 'Systemic Lupus Erythematosus'
}

FINAL_FEATURE_NAMES = [
        'age', 'ESR', 'CRP', 'RF', 'antiCCP', 'C3', 'C4',
        'gender', 'HLA-B27', 'ANA', 'antiRo', 'antiLa', 'antiDsDNA', 'antiSm'
]

# Global require imputation flag
if 'requires_imputation' not in st.session_state:
    st.session_state.requires_imputation = False

# Functions
# Loading files/models function
@st.cache_resource
def load_resources():
    numerical_imputer = joblib.load('numerical_imputer.joblib')
    categorical_imputer = joblib.load('categorical_imputer.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')

    # p1 = joblib.load('preprocessor_1.joblib')
    # p2 = joblib.load('preprocessor_2.joblib')
    # knn = joblib.load('knn_imputer.joblib')

    # Not using this anymore
    # synthesizer = load_synthesizer('autoimmune_rheumatic_diagnosis_synthesizer_27-12-2025-18-18.pkl')

    lgbmc = joblib.load('best_lgbmc_model.joblib')

    # Initialise the SHAP TreeExplainer
    explainer = shap.TreeExplainer(lgbmc)

    X_sample = shap.sample(joblib.load('X_train_balanced_final.joblib'), 1000)

    # Calculate global SHAP values
    global_shap_values = explainer.shap_values(X_sample)

    global_shap_explanation = explainer(X_sample)

    GEMINI_API_KEY = st.secrets["gemini_api_key"]

    gemini = genai.Client(api_key=GEMINI_API_KEY)

    # Load the LIME explainer using dill
    with open('latest_lime_explainer_final.pkl', 'rb') as file:
        lime_explainer = dill.load(file)

    # KNN imputer for LIME
    # knn_lime = joblib.load('knn_lime_imputer.joblib')

    return numerical_imputer, categorical_imputer, ohe, lgbmc, explainer, X_sample, global_shap_values, global_shap_explanation, GEMINI_API_KEY, gemini, lime_explainer

# Immediately load the models
numerical_imputer, categorical_imputer, ohe, model, explainer, X_sample, global_shap_values, global_shap_explanation, GEMINI_API_KEY, gemini, lime_explainer = load_resources()

# Transforming input data function
def transform_data(raw_input_data):
  """
  Returns final scaled row.
  """

  # Convert raw data from dictionary form to DataFrame form
  loaded_row = pd.DataFrame([raw_input_data])
  imputed_row = pd.DataFrame()

  # FEATURE SCALING
  # # Features to scale (non-binary)
  # continuous_features = ['age', 'ESR', 'CRP', 'RF', 'antiCCP', 'C3', 'C4']
  #
  # # Get binary features
  # binary_features = loaded_row.columns.drop(continuous_features).tolist()
  #
  # # Transform the data
  # scaled_row_array = preprocessor_1.transform(loaded_row)
  #
  # new_column_order = continuous_features + binary_features
  #
  # scaled_row = pd.DataFrame(scaled_row_array, columns=new_column_order)

  # IMPUTING MISSING VALUES
  # Create imputed row here earlier in case there are no missing values
  imputed_row = loaded_row.copy()

  # torch.cuda_is_available() in place of False if GPU is available
  if imputed_row.isna().values.any() and False:
    imputed_row = imputed_row
    # st.session_state.requires_imputation = True
    #
    # row = imputed_row.copy()
    #
    # # Drop the columns where values are missing
    # known_features = row.dropna()

    # Export the values
    # known_features_dict = known_features.to_dict()

    # Create the sampling condition
    # imputation_condition = Condition(
    #     num_rows=1,
    #     column_values=known_features_dict
    # )

    # Create the imputed rows
    # Uncomment line below when using GPU
    # imputed_row = synthesizer.sample_from_conditions([imputation_condition])

  elif imputed_row.isna().values.any():
    st.session_state.requires_imputation = True

    # Impute using the numerical (mean) and categorical (mode) imputer
    # Define the categorical and numerical columns
    numerical_columns = ['age', 'ESR', 'CRP', 'RF', 'antiCCP', 'C3', 'C4']
    categorical_columns = ['gender', 'HLA-B27', 'ANA', 'antiRo', 'antiLa', 'antiDsDNA', 'antiSm']

    # Transform the row using numerical and categorical imputers (mean and mode)
    imputed_row[numerical_columns] = numerical_imputer.transform(imputed_row[numerical_columns])
    imputed_row[categorical_columns] = categorical_imputer.transform(imputed_row[categorical_columns])

    # Create a mapping dictionary
    mapping = {"Negative": 0, "Positive": 1}

    # Apply to the categorical columns
    for column in categorical_columns:
        imputed_row[column] = imputed_row[column].replace(mapping)

    # raise ValueError(imputed_row[categorical_columns])

    # Use One-Hot Encoding for encoding categorical features
    # imputed_row_encoded_array = ohe.transform(imputed_row[categorical_columns])
    # imputed_row_encoded = pd.DataFrame(imputed_row_encoded_array, columns=ohe.get_feature_names_out(categorical_columns))

    # Reset the encoded column names into its original column names
    # imputed_row_encoded.columns = [column.split('_')[0] for column in imputed_row_encoded.columns]

    # Concatenate the encoded categorical columns back with the numerical columns
    # concatenated_row = pd.concat([imputed_row[numerical_columns].reset_index(drop=True), imputed_row_encoded.reset_index(drop=True)], axis=1)
    concatenated_row = imputed_row.copy()


    # imputed_lime_row_array = knn_lime_imputer.transform(loaded_row)

    # Convert the imputed features back and its column names to a DataFrame (because imputer returns a NumPy array, not a DataFrame)
    # imputed_row = pd.DataFrame(imputed_row_array, columns=scaled_row.columns)
    # imputed_lime_row = pd.DataFrame(imputed_lime_row_array, columns=scaled_row.columns)

    # Round the values of the binary columns
    # for column in binary_columns:
    #     imputed_row[column] = np.round(imputed_row[column])
    #     imputed_row[column] = imputed_row[column].astype(int)

  else:
    st.session_state.requires_imputation = False
    concatenated_row = imputed_row.copy()

        # imputed_lime_row[column] = np.round(imputed_lime_row[column])
        # imputed_lime_row[column] = imputed_lime_row[column].astype(int)

  # FEATURE ENGINEERING
  # Get back the StandardScaler of the first feature scaling
  # initial_scaler = preprocessor_1.named_transformers_['cont_scaler']

  # Get the indices for ESR and CRP
  # ESR_index = 1
  # CRP_index = 2

  # Get ESR and CRP's mean and scale
  # ESR_mean = initial_scaler.mean_[ESR_index]
  # ESR_scale = initial_scaler.scale_[ESR_index]
  #
  # CRP_mean = initial_scaler.mean_[CRP_index]
  # CRP_scale = initial_scaler.scale_[CRP_index]

  # Clinical thresholds for inflammation status
  # ESR_clinical_threshold = 20.0
  # CRP_clinical_threshold = 5.0

  # Calculating the Z-score of the threshold (because ESR and CRP in balanced DFs are already scaled)
  # Z_ESR_threshold = (ESR_clinical_threshold - ESR_mean) / ESR_scale
  # Z_CRP_threshold = (CRP_clinical_threshold - CRP_mean) / CRP_scale

  # Creating the features
  # lime_row = imputed_row.copy()

  # for row in [imputed_row]:
  #   # Ratio of acute phase reactants
  #   row['CRP_ESR_ratio'] = row['CRP'] / row['ESR']
  #   row['RF_antiCCP_ratio'] = row['RF'] / row['antiCCP']
  #   row['C3_C4_ratio'] = row['C3'] / row['C4']
  #
  #   # Immunological activity scores
  #   # Extractable Nuclear Antigen (ENA) Count
  #   row['ena_count'] = row['antiRo'] + row['antiLa'] + row['antiSm']
  #   row['systemic_autoantibody_count'] = row['ANA'] + row['antiDsDNA'] + row['antiSm']
  #   row['rf_antibody_score'] = row['RF'] * (row['antiCCP'] + 1)
  #
  #   # Interaction and status features
  #   if row is imputed_row:
  #     row['inflammation_status'] = np.where(
  #       (row['ESR'] > Z_ESR_threshold) | (row['CRP'] > Z_CRP_threshold),
  #       1, # Value to put if True
  #       0  # Value to put if False
  #     )
  #   else:
  #     row['inflammation_status'] = np.where(
  #       (row['ESR'] > ESR_clinical_threshold) | (row['CRP'] > CRP_clinical_threshold),
  #       1, # Value to put if True
  #       0  # Value to put if False
  #     )
  #
  #   row['spondyloarthropathy_risk'] = row['HLA_B27'] * row['age']

  # imputed_row = imputed_row.mask(np.isinf(imputed_row), 0)
  # imputed_lime_row = imputed_lime_row.mask(np.isinf(imputed_lime_row), 0)

  # RE-SCALING FEATURES
  # Features to scale (non-binary)
  # final_continuous_features = ['age', 'ESR', 'CRP', 'RF', 'antiCCP', 'C3', 'C4', 'CRP_ESR_ratio', 'RF_antiCCP_ratio', 'C3_C4_ratio', 'ena_count', 'systemic_autoantibody_count', 'rf_antibody_score', 'spondyloarthropathy_risk']

  # Get binary features
  # final_binary_features = imputed_row.columns.drop(final_continuous_features).tolist()

  # new_column_order_final = final_continuous_features + final_binary_features

  # imputed_row = imputed_row[new_column_order_final]
  # scaled_row_array_final = preprocessor_2.transform(imputed_row)

  # Reassemble the DataFrame
  # Scaled features first, then passthrough features (continuous + binary)
  # final_scaled_row = pd.DataFrame(scaled_row_array_final, columns=new_column_order_final)

  # imputed_lime_row = imputed_lime_row[new_column_order_final]

  # final_scaled_row_array = final_scaled_row.values
  # imputed_lime_row_array = imputed_lime_row.values

  return concatenated_row, "no_longer_required"

# def transform_lime_data(raw_input_data):
#     """
#     Specifically for LIME: Transforms raw input into the 22-feature unscaled
#     format required for the LIME graph labels.
#     """
#     # knn_imputer_column_format = knn_lime_imputer.feature_names_in_
#
#     # Handle dictionary input (Streamlit form with 14 features)
#     if isinstance(raw_input_data, dict):
#         df = pd.DataFrame([raw_input_data], columns=knn_imputer_column_format)
#
#     # Handle NumPy Array Input (LIME - 22 features)
#     else:
#         data_array = np.array(raw_input_data)
#         if data_array.ndim == 1:
#             data_array = data_array.reshape(1, -1)
#
#         # Assign names based on column count to avoid KeyError
#         if data_array.shape[1] == 14:
#             raw_feature_names = ['age', 'gender', 'ESR', 'CRP', 'RF', 'antiCCP',
#                                  'HLA_B27', 'ANA', 'antiRo', 'antiLa', 'antiDsDNA',
#                                  'antiSm', 'C3', 'C4']
#             df = pd.DataFrame(data_array, columns=raw_feature_names)
#         elif data_array.shape[1] == 22:
#             # Perturbation from LIME engine
#             df = pd.DataFrame(data_array, columns=FINAL_FEATURE_NAMES)
#         else:
#             raise ValueError(
#                 f"Unexpected column count: {data_array.shape[1]}. Expected 14 (Streamlit form) or 22 columns (LIME examples).")
#
#     if df.isna().values.any():
#         df_columns = df.columns
#         imputed_array = knn_lime_imputer.transform(df)
#         df = pd.DataFrame(imputed_array, columns=df_columns)
#
#     # Recalculate engineered features (Ratios/Counts)
#     df['CRP_ESR_ratio'] = df['CRP'] / df['ESR']
#     df['RF_antiCCP_ratio'] = df['RF'] / df['antiCCP']
#     df['C3_C4_ratio'] = df['C3'] / df['C4']
#     df['ena_count'] = df['antiRo'] + df['antiLa'] + df['antiSm']
#     df['systemic_autoantibody_count'] = df['ANA'] + df['antiDsDNA'] + df['antiSm']
#     df['rf_antibody_score'] = df['RF'] * (df['antiCCP'] + 1)
#     df['spondyloarthropathy_risk'] = df['HLA_B27'] * df['age']
#
#     # Use clinical thresholds for status
#     df['inflammation_status'] = np.where(
#         (df['ESR'] > 20.0) | (df['CRP'] > 5.0), 1, 0
#     )
#
#     for column in ['gender', 'HLA_B27', 'ANA', 'antiRo', 'antiLa', 'antiDsDNA', 'antiSm', 'inflammation_status']:
#         df[column] = df[column].astype(int)
#
#     # Reorder columns to match FINAL_FEATURE_NAMES exactly
#     df_lime = df[FINAL_FEATURE_NAMES].copy()
#
#     # Clean up any infinity values from division
#     df_lime = df_lime.mask(np.isinf(df_lime), 0)
#
#     return df_lime.values  # Returns the 22-feature unscaled array

# Retrieving the autoimmune rheumatic disease class prediction
def get_prediction(transformed_data):
    """
    Takes the transformed, runs it through the CatBoostClassifier (CBC) model, and returns probabilities and the top class index.
    """
    # CatBoost model's predict_proba() function returns a 2D array for 2D inputs
    # Even for one row (individual prediction), it will return shape of (1, n_classes)
    probabilities = model.predict_proba(transformed_data)[0]
    top_class_idx = np.argmax(probabilities)

    return probabilities, top_class_idx

# Display the autoimmune rheumatic disease prediction results
def display_results(probabilities, top_class_idx):
    """
    Renders the autoimmune rheumatic disease prediction results in the Streamlit UI.
    """

    predicted_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[top_class_idx]
    confidence = probabilities[top_class_idx]

    st.header("Autoimmune Rheumatic Diagnostic Prediction Results")

    # Imputation performed warning
    if st.session_state.get('requires_imputation', True):
        st.warning(f":red-background[**Note**] You have left at least one field blank in the form. The value for the blank field has been imputed accordingly, but it is best to fill in the value for the most accurate results.", icon=':material/exclamation:')

    # 1. Main callout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Most likely diagnosis: **{predicted_disease}**")
        st.progress(float(confidence), text=f"Confidence: {confidence * 100:.1f}%")
    with col2:
        st.metric(label="Highest Probability", value=f"{confidence * 100:.1f}%")

    st.divider()

    # 2. Probability breakdown table and chart
    st.subheader("Probability distribution across all conditions")

    # Create a DataFrame for display
    results_df = pd.DataFrame({
        "Condition": AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES,
        "Probability (%)": [round(p * 100, 2) for p in probabilities]
    }).sort_values(by="Probability (%)", ascending=False)

    # Use columns to show the table and chart side-by-side
    chart_col, table_col = st.columns([3, 2])

    with chart_col:
        # Simple horizontal bar chart
        st.bar_chart(results_df, x="Condition", y="Probability (%)", color="#007bff", horizontal=True, sort="-Probability (%)")

    with table_col:
        st.dataframe(results_df, hide_index=True, width='stretch')

    # 3. Warning to the clinician/user regarding the limitations of this Streamlit application
    st.warning(
        ":red-background[**Note**] This application is a diagnostic clinical decision support aid generated by a ***machine learning model***. Clinical correlation and confirmation by a rheumatologist or another relevant specialist is **MANDATORY**!")

# Convert a Matplotlib figure to a PIL image
def get_pil_image_from_matplotlib(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format='png')
    buffer.seek(0)
    return Image.open(buffer)

# Convert a Plotly figure to a PIL image
def get_pil_image_from_plotly(figure):
    width = 1200
    height = 1000

    image_bytes = figure.to_image(format="png", width=width, height=height, scale=2)
    return Image.open(BytesIO(image_bytes))

def explain_graph_with_llm(is_plotly, figure, figure_context=None):
    if is_plotly:
        st.session_state['pending_explainer_image'] = get_pil_image_from_plotly(figure)
    else:
        st.session_state['pending_explainer_image'] = get_pil_image_from_matplotlib(figure)

    st.session_state['pending_figure_context'] = figure_context
    st.rerun()

def display_global_interpretability(global_shap_values, global_shap_explanation, feature_names, X_sample, top_class_idx):
    st.divider()
    st.header("Global Model Insights")
    st.write("These charts show which features influence the prediction of the machine learning model across all patients.")

    st.subheader("Differential Analysis")
    st.write(
        "Select a condition to see the global influences of features for a specific condition.")

    # Default to the top prediction, but have an option for choosing other conditions
    global_selected_class_name = st.selectbox(
        "View analysis for:",
        options=AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES,
        index=int(top_class_idx),
        key='global_shap_disease_selectbox'
    )

    # Get the index of the selected condition
    analysis_idx = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES.index(global_selected_class_name)
    top_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[top_class_idx]
    selected_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[analysis_idx]

    if analysis_idx == top_class_idx:
        st.write(f"The predicted condition, :yellow-background[{top_disease}], is currently :green-background[**selected**].")
    else:
        st.write(f"The predicted condition, :yellow-background[{top_disease}], is currently :red-background[***not selected***].")

    st.markdown("<a name='global_shap'></a>", unsafe_allow_html=True)
    tabs = st.tabs(["Feature Importance", "Feature Impact (Beeswarm)", "Feature Dependence"])

    with tabs[0]:
        # Calculate global average absolute SHAP values
        st.write(f"The feature importance graph shows how important a feature is in a prediction, :yellow-background[*regardless of a specific condition*].")
        figure_context = "A global SHAP feature importance graph that shows how important a feature is in a prediction, regardless of a specific condition."

        shap_abs_mean = np.abs(np.array(global_shap_values)).mean(axis=(0, 2))

        df_global_shap = pd.DataFrame({
            'Feature': feature_names,
            'Mean Absolute SHAP': shap_abs_mean
        }).sort_values(by='Mean Absolute SHAP', ascending=True)

        fig_global = px.bar(
            df_global_shap,
            x='Mean Absolute SHAP',
            y='Feature',
            orientation='h',
            title='Overall Feature Importance',
            color_discrete_sequence=['#007bff']
        )

        st.plotly_chart(fig_global, width='stretch')

        if st.button(
            label="Explain this graph for me",
            key='global_feature_importance_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Global Feature Importance graph',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            explain_graph_with_llm(True, fig_global, figure_context=figure_context)

    with tabs[1]:
        # Beeswarm plot (using Matplotlib)
        st.write(f"The beeswarm graph shows how higher or lower values of a feature generally influence the model to lean :green-background[towards] or :red-background[away] from predicting the :yellow-background[{selected_disease}] condition.")
        figure_context = f"A SHAP global beeswarm graph that shows how higher or lower values of a feature generally influence the model to lean towards or away from predicting the {selected_disease} condition."

        # Get the SHAP values
        beeswarm_shap_values = explainer(X_sample)

        # Index the SHAP values: [all rows, all features, highest probability class]
        shap.plots.beeswarm(beeswarm_shap_values[:, :, analysis_idx], max_display=22)
        # Pass the plot to Streamlit
        st.pyplot(plt.gcf())
        beeswarm_plot = plt.gcf()
        plt.close()

        if st.button(
            label="Explain this graph for me",
            key='global_beeswarm_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Global Feature Impact graph',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            explain_graph_with_llm(False, beeswarm_plot, figure_context=figure_context)

    with tabs[2]:
        # Feature dependence plot (also using Matplotlib)
        st.write(f"The feature dependence graph shows how the risk of the :yellow-background[{selected_disease}] condition :green-background[rises] or :red-background[falls] along the (scaled with mean of 0 and standard deviation of 1) values of a specific feature.")
        st.write("Select a feature to see how its value affects the prediction probability.")

        # Select box for selected feature
        selected_feature = st.selectbox("Select feature to analyse:", feature_names, key="global_dependence_feature")

        # Checkbox to toggle secondary feature interaction colour effect
        show_secondary_interaction = st.checkbox(f"Colour the effects between {selected_feature} and the feature it has the strongest interaction with", value=True, key="toggle_interaction")

        if show_secondary_interaction:
            figure_context = f"A SHAP global feature dependence graph that shows how the risk of the {selected_disease} condition rises or falls along the (scaled with mean of 0 and standard deviation of 1) values of {selected_feature}. Note that a second feature is included where it has the strongest interaction effects with {selected_feature}."
        elif not show_secondary_interaction:
            figure_context = f"A SHAP global feature dependence graph that shows how the risk of the {selected_disease} condition rises or falls along the (scaled with mean of 0 and standard deviation of 1) values of {selected_feature}."

        # Generate feature dependence (scatter) plot
        # Filter to the predicted disease class
        disease_explanation = global_shap_explanation[:, :, analysis_idx]

        if show_secondary_interaction:
            shap.plots.scatter(disease_explanation[:, selected_feature], color=disease_explanation)
        else:
            shap.plots.scatter(disease_explanation[:, selected_feature])

        st.pyplot(plt.gcf())
        feature_dependence_plot = plt.gcf()
        plt.close()

        if st.button(
            label="Explain this graph for me",
            key='global_feature_dependence_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Global Feature Dependence graph',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            explain_graph_with_llm(False, feature_dependence_plot, figure_context=figure_context)

def display_local_shap_interpretability(transformed_data, top_class_idx):
    st.divider()
    st.header("Patient-Specific Analysis")
    st.subheader("Local Shapley Additive Explanations (SHAP)")

    st.subheader("Differential Analysis")
    st.write("Select a condition to see why the machine learning model supported or discounted it specifically for this patient.")

    # Default to the top prediction, but have an option for choosing other conditions
    local_shap_selected_class_name = st.selectbox(
        "View analysis for:",
        options=AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES,
        index=int(top_class_idx),
        key='local_shap_disease_selectbox'
    )

    # Get the index of the selected condition
    analysis_idx = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES.index(local_shap_selected_class_name)

    global top_disease
    top_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[top_class_idx]

    selected_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[analysis_idx]

    if analysis_idx == top_class_idx:
        st.write(f"The predicted condition, :yellow-background[{top_disease}], is currently :green-background[**selected**].")
    else:
        st.write(f"The predicted condition, :yellow-background[{top_disease}], is currently :red-background[***not selected***].")
    # Calculate SHAP for the input patient
    # This returns a list of arrays (one per class)
    patient_shap_values = explainer.shap_values(transformed_data)

    # Extract values for the predicted class and flatten it to 1D
    # Take the array for analysis_idx and flatten it to ensure the shape is (22,) which is 1D, and not (1, 22), which is 2D
    # Convert to a numpy array to check shape
    patient_shap_array = np.array(patient_shap_values)

    # If the shape is 3D (rows, features, classes), format to:
    # All rows (only 1 or the input patient), all features, predicted class
    global current_patient_shap_1d
    if len(patient_shap_array.shape) == 3:
        current_patient_shap_1d = patient_shap_array[0, :, analysis_idx]
    else:
        # If list style, just flatten the array given the predicted class
        current_patient_shap_1d = patient_shap_array[analysis_idx].flatten()

    base_values = explainer.expected_value

    if isinstance(base_values, (list, np.ndarray)) and len(base_values) > analysis_idx:
        base_value = base_values[analysis_idx]
    else:
        base_value = base_values  # Fallback if it's a scalar

    tabs = st.tabs(["Bar Plot", "Force Plot", "Waterfall Plot"])

    # Bar plot
    with tabs[0]:
        if analysis_idx == top_class_idx:
            st.write(f"Top features driving the prediction :green-background[***for***] :yellow-background[{selected_disease}]:")
            figure_context = f"A local SHAP bar plot, showing the top features driving the prediction for {selected_disease}."
        else:
            st.write(f"Top features driving the prediction :red-background[***away from***] :yellow-background[{selected_disease}]:")
            figure_context = f"A local SHAP bar plot, showing the top features driving the prediction away from {selected_disease}."

        df_bar = pd.DataFrame({
            'Feature': FINAL_FEATURE_NAMES,
            'SHAP Value': current_patient_shap_1d
        })

        df_bar['Absolute SHAP Value'] = df_bar['SHAP Value'].abs()
        df_bar = df_bar.sort_values(by='Absolute SHAP Value', ascending=False)

        fig_bar = px.bar(
            df_bar,
            x='SHAP Value',
            y='Feature',
            orientation='h',
            color=df_bar['SHAP Value'] > 0,
            color_discrete_map={True: '#636efa', False: '#ef553b'},
            labels={'SHAP Value': 'Impact Score'}
        )
        fig_bar.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, width='stretch')

        if st.button(
            label="Explain this graph for me",
            key='local_shap_bar_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Local SHAP Bar Plot',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            explain_graph_with_llm(True, fig_bar, figure_context=figure_context)

    # Force plot
    with tabs[1]:
        st.write(f"This force plot shows how each feature from the patient pushed stronger or weaker towards the prediction of the :yellow-background[{selected_disease}] condition.")

        figure_context = f"A local SHAP force plot that shows how much each feature from the patient pushed stronger or weaker towards the prediction of the {selected_disease} condition."

        # HTML wrapper function because force plot is done in JavaScript
        def call_html_force_plot(plot, height=None):
            # Wrap the JS and the plot HTML in a format Streamlit can render
            shap_html = f"""
                    <head>
                        {shap.getjs()}
                        <style>
                            body {{
                                background-color: white !important;
                                margin: 0;
                                padding: 10px;
                                font-family: sans-serif;
                            }}
                        </style>
                    </head>
                    <body>
                        <div style="background-color: white; padding: 10px; border-radius: 5px;">
                            {plot.html()}
                        </div>
                    </body>
                    """
            components.html(shap_html, height=height)

        # Generate the force plot
        # Use flatten() for the data to ensure it is a 1D vector of values
        force_plot = shap.force_plot(
            base_value,
            current_patient_shap_1d,
            transformed_data.values.flatten(),
            feature_names=FINAL_FEATURE_NAMES,
            matplotlib=False  # Set to False for the interactive JS version for Streamlit
        )

        # Call the help function to render the force plot
        call_html_force_plot(force_plot, height=200)

        if st.button(
            label="Explain this graph for me",
            key='local_shap_force_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Local SHAP Force Plot',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            # Have to use Matplotlib version to pass to LLM
            shap.force_plot(
                base_value,
                current_patient_shap_1d,
                transformed_data(),
                feature_names=FINAL_FEATURE_NAMES,
                show=False,
                matplotlib=True  # Set to True for the LLM explainer version
            )

            force_plot_figure = plt.gcf()
            force_plot_figure.set_size_inches(12, 3)

            explain_graph_with_llm(False, force_plot_figure, figure_context=figure_context)

    # Waterfall plot
    with tabs[2]:
        st.write(f"The waterfall plot shows how the patient's data starts from patient average to the input patient's particular prediction for the :yellow-background[{selected_disease}] condition.")
        figure_context = f"A local SHAP waterfall plot that shows how the patient's data starts from patient average to the input patient's particular prediction for the {selected_disease} condition."

        # Use an Explanation object
        # Create a temporary Explanation object for the input patient
        explanation = shap.Explanation(
            values=current_patient_shap_1d,
            base_values=base_value,
            data=transformed_data.values.flatten(),
            feature_names=FINAL_FEATURE_NAMES
        )

        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig_waterfall)
        waterfall_plot_figure = plt.gcf()
        plt.close()

        if st.button(
            label="Explain this graph for me",
            key='local_shap_waterfall_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Local SHAP Waterfall Plot',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
        ):
            explain_graph_with_llm(False, waterfall_plot_figure, figure_context=figure_context)

def lime_predict_proba(unscaled_numpy_array):
    """
    Wrapper for LIME to bridge unscaled perturbations and the scaled model.
    """
    # Convert LIME's numpy output back to a dictionary or DataFrame
    # LIME sends a 2D array, so process row by row or as a batch
    probabilities = []

    for row in unscaled_numpy_array:
        row_dict = dict(zip(FINAL_FEATURE_NAMES[:14], row))

        retransformed_data_df, _ = transform_data(row_dict)
        probs = model.predict_proba(retransformed_data_df)
        probabilities.append(probs[0])

    return np.array(probabilities)

def display_local_lime_interpretability(transformed_data, top_class_idx):
    st.divider()
    st.subheader("Local Interpretable Model-agnostic Explanations (LIME)")

    st.subheader("Differential Analysis")
    st.warning(f":red-background[**Note**] LIME graphs are computationally expensive and may take a few seconds to load. Generated graphs are saved until a prediction for another patient is made.", icon=':material/search_activity:')
    st.write("Select a condition to see why the machine learning model supported or discounted it specifically for this patient.")

    # Default to the top prediction, but have an option for choosing other conditions
    local_lime_selected_class_name = st.selectbox(
        "View analysis for:",
        options=AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES,
        index=int(top_class_idx),
        key='local_lime_disease_selectbox'
    )

    # Get the index of the selected condition
    lime_analysis_idx = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES.index(local_lime_selected_class_name)

    st.session_state["last_lime_analysis_idx"] = lime_analysis_idx

    global top_lime_disease
    top_lime_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[top_class_idx]
    selected_lime_disease = AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[lime_analysis_idx]

    if lime_analysis_idx == top_class_idx:
        st.write(
            f"The predicted condition, :yellow-background[{top_lime_disease}], is currently :green-background[**selected**].")
    else:
        st.write(
            f"The predicted condition, :yellow-background[{top_lime_disease}], is currently :red-background[***not selected***].")

    cache_lime_explainer_key = f"lime_explainer_{lime_analysis_idx}"
    cache_key = f"lime_figure_{lime_analysis_idx}"
    cache_figure_context_key = f"lime_figure_context_{lime_analysis_idx}"

    if cache_key not in st.session_state:
        with st.spinner(f"Generating LIME graph for the {selected_lime_disease} condition, please wait..."):
            # Initialise LIME explainer
            lime_explanation = lime_explainer.explain_instance(
                data_row=transformed_data.values.flatten(),
                predict_fn=lime_predict_proba,
                num_features=14,
                labels=[lime_analysis_idx],
                num_samples=2000
            )

            # Extract the list of (feature_description, weight) for the predicted class
            # lime_explanation.as_list() returns tuples such as ('CRP > 5.0', 0.15)
            lime_list = lime_explanation.as_list(label=lime_analysis_idx)

            # Create a DataFrame for the Plotly graph
            lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Weight'])

            # Decide title based on selected disease for analysis
            if lime_analysis_idx == top_class_idx:
                title = f"LIME Explanation: Why the Model Predicted Towards the {selected_lime_disease} Condition"
                figure_context = f"A local LimeTabularExplainer explain instance bar graph, showing the features driving the prediction for {selected_lime_disease}."
            else:
                title = f"LIME Explanation: Why the Model Predicted Away from the {selected_lime_disease} Condition"
                figure_context = f"A local LimeTabularExplainer explain instance bar graph, showing the features driving away the prediction for {selected_lime_disease}."

            # Generate the Plotly horizontal bar chart
            lime_figure = px.bar(
                lime_df,
                x='Weight',
                y='Feature',
                orientation='h',
                color=lime_df['Weight'] > 0,
                color_discrete_map={True: '#636efa', False: '#ef553b'}, # Blue colour for positive contribution, red colour for negative contribution
                title=title
            )

            lime_figure.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Influence on Prediction",
                template="plotly_white"
            )

            # Store Plotly figure in session state
            st.session_state[cache_lime_explainer_key] = lime_explanation
            st.session_state[cache_key] = lime_figure
            st.session_state[cache_figure_context_key] = figure_context

            st.session_state["last_lime_explanation"] = lime_explanation
            st.session_state["last_lime_analysis_idx"] = lime_analysis_idx

    # Display the chart in Streamlit
    lime_figure = st.session_state[cache_key]
    st.plotly_chart(lime_figure, use_container_width=True)

    figure_context = st.session_state[cache_figure_context_key]

    # Explain LIME graph button
    if st.button(
            label="Explain this graph for me",
            key='local_lime_explainer_button',
            help='Use the LLM Clinical Assistant to explain the Local LIME Bar Plot',
            type='primary',
            icon=':material/chat_add_on:',
            width='stretch'
    ):
        explain_graph_with_llm(True, lime_figure, figure_context=figure_context)

def prepare_shap_summary_for_gemini(current_patient_shap_1d, feature_names):
    # Create a DataFrame to sort impacts
    df = pd.DataFrame({'Feature': feature_names, 'Impact': current_patient_shap_1d})
    df['Abs_Impact'] = df['Impact'].abs()
    top_drivers = df.sort_values(by='Abs_Impact', ascending=False).head(10)

    summary = ""
    for _, row in top_drivers.iterrows():
        direction = "Positive (Supporting)" if row['Impact'] > 0 else "Negative (Opposing)"
        summary += f"- {row['Feature']}: {direction} SHAP impact of {row['Impact']:.4f}\n"
    return summary

def prepare_lime_summary_for_gemini(lime_explanation, label_idx):
    """
    Extracts the feature descriptions and weights from LIME to provide textual context to the Gemini LLM.
    """
    # Extract the list of (feature_description, weight)
    lime_list = lime_explanation.as_list(label=label_idx)

    summary = "LIME Clinical Breakdown:\n"
    for feature_description, weight in lime_list:
        # Clean the mu symbol if it exists in the raw strings
        clean_description = feature_description.replace('\u03bc', 'value')
        direction = "Supporting" if weight > 0 else "Opposing"
        summary += f"- {clean_description}: {direction} impact (Weight: {weight:.4f})\n"

    return summary

def run_chatbot_interface(predicted_disease, probabilities, feature_names, current_shap_1d, explainer_image=None, figure_context=None):
    # Sidebar chat window
    with st.sidebar:
        st.markdown(
            """
        <style>
        .st-emotion-cache-155jwzh {
            overflow: hidden;
        }
            .st-emotion-cache-10p9htt {
            height: 1rem;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Header and clear chat layout
        header_column, button_column = st.columns(spec=[0.7, 0.3], gap='small', vertical_alignment='center', width='stretch')

        with header_column:
            st.header(":material/forum: LLM Clinical Assistant")

        with button_column:
            st.write('')
            # if st.button('**Clear Chat**', help="Delete all messages and reset the AI's context", icon=':material/delete:', type='primary', width='stretch'):



            # Initialise the confirmation state if it has not existed yet
            if 'confirm_delete_chat' not in st.session_state:
                st.session_state.confirm_delete_chat = False

            # Show the main Clear Chat button if not in confirmation mode
            if not st.session_state.confirm_delete_chat:
                if st.button('**Clear Chat**', help="Delete all messages and reset the Clinical Assistant's context", icon=':material/delete:', type='primary', width='stretch'):
                    st.session_state.confirm_delete_chat = True
                    st.rerun()
            else:
                # We are in confirmation mode: show either Yes and No
                st.write("**:red[Are you sure?]**")
                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("Yes", type="primary", key="confirm_yes_delete_chat", width='stretch', help="This action cannot be undone!"):
                        # Reset all messages currently in the session state (wiping visual messages)
                        st.session_state.messages = []

                        if "chat_session" in st.session_state:
                            del st.session_state.chat_session

                        # Set the confirm delete chat toggle back to False
                        st.session_state.confirm_delete_chat = False

                        # Rerun the chatbot to show the changes
                        st.rerun()

                with col_no:
                    if st.button("No", key="confirm_no_delete_chat", width='stretch', help='Cancel this action'):
                        # Set the confirm delete chat toggle back to False
                        st.session_state.confirm_delete_chat = False

                        # Refresh the chatbot page
                        st.rerun()

        st.success(f":violet-background[**Context**] Currently analysing the diagnosis of ***{predicted_disease}*** in the patient", icon=':material/search_activity:')
        # st.info(f":orange-background[**Instructions**] Use the LLM to query about what the SHAP values obtained means, how the SHAP values explain influences in patient features that led to this prediction, and more.", icon=':material/chat_info:')
        st.warning(f":red-background[**Note**] The responses generated from this chat are produced by generative artificial intelligence (GenAI). Please take due diligence in validating the information generated against verified medical knowledge.", icon=':material/exclamation:')

        # Container for prompts and responses
        chat_container = st.container(height=500)

        # If a chat session has not started yet, start it and keep it in the session
        if "chat_session" not in st.session_state:
            # Intitialise history of messages
            history = []

            # Append any messages into the history
            for message in st.session_state.messages:
                role = "model" if message["role"] == "assistant" else "user"
                history.append({"role": role, "parts": [{"text": message["content"]}]})

            st.session_state.chat_session = gemini.chats.create(
                model="gemini-3-flash-preview",
                config=GenerateContentConfig(
                    system_instruction=[
                        "You are an AI for an autoimmune rheumatology disease diagnosis machine learning model, as part of a clinical decision support system.",
                        "The possible conditions to be classified in the model are either: Ankylosing Spondylitis, Normal, Psoriatic Arthritis, Reactive Arthritis, Rheumatoid Arthritis, Sjögren\'s Syndrome, or Systemic Lupus Erythematosus.",
                        """
                        Some of the features of the model are feature-engineered models. The definition of the engineered features (for your understanding and responses in answers) are:
                        dataset['CRP_ESR_ratio'] = dataset['CRP'] / dataset['ESR']
                        dataset['RF_antiCCP_ratio'] = dataset['RF'] / dataset['antiCCP']
                        dataset['C3_C4_ratio'] = dataset['C3'] / dataset['C4']
                        
                        dataset['ena_count'] = dataset['antiRo'] + dataset['antiLa'] + dataset['antiSm']
                        dataset['systemic_autoantibody_count'] = dataset['ANA'] + dataset['antiDsDNA'] + dataset['antiSm']
                        dataset['rf_antibody_score'] = dataset['RF'] * (dataset['antiCCP'] + 1)
                        
                        if dataset is [X_train_balanced, X_test_balanced]:
                            dataset['inflammation_status'] = np.where(
                                (dataset['ESR'] > Z_ESR_threshold) | (dataset['CRP'] > Z_CRP_threshold),
                                1, # Value to put if True
                                0  # Value to put if False
                            )
                        else:
                            dataset['inflammation_status'] = np.where(
                                (dataset['ESR'] > ESR_clinical_threshold) | (dataset['CRP'] > CRP_clinical_threshold),
                                1, # Value to put if True
                                0  # Value to put if False
                            )

                        dataset['spondyloarthropathy_risk'] = dataset['HLA_B27'] * dataset['age']
                        """,
                        "Usually, the initial prompt the user sends will be of the predicted autoimmune rheumatic disease, with its confidence in %.",
                        "The prediction model uses a machine learning model with global and local SHAP graphs to add to the explainability factor of the prediction.",
                        "A summary of the key local SHAP drivers and local LIME clinical threshold context of the prediction is typically also sent in a user's initial prompt.",
                        "As the computer-science-to-natural-language translation layer, you will be the one to translate the SHAP and LIME graphs (that are difficult to interpret by clinicians and others) into natural-language terms that they can understand.",
                        "Answer briefly and be clinically accurate as much as possible. It is essential that you maximise interpretability of the prediction and explainability graphs to support the clinician and their further decisions/actions to take for the patient.",
                        "Refer to specific lab values when possible."
                    ]
                ),
                history=history
            )

        # In the chat container, display chat messages according to their roles from history on app rerun
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):  # Display the chat message according to the role
                    st.markdown(message["content"])     # Display the text prompt
                    if "images" in message:             # If there were any images in the prompt
                        for image in message["images"]: # For each image in the prompt
                            st.image(image)             # Display the image

        # React to user input
        prompt = st.chat_input(placeholder="Ask about this diagnosis prediction...",
                                accept_file="multiple",
                                file_type=["jpg", "jpeg", "png"])

        user_text = ""
        user_files = []

        # explainer_image is not None when a user presses a button to explain a graph
        # That image is passed into a run chatbot interface call with explainer_image as the image
        if explainer_image is not None:
            user_text = "Explain this graph for me"
            user_files.append(explainer_image)
        # There was no explainer image passed and there was text in the chat input box
        elif prompt:
            # Extract the text and files from the chat_input (dictionary)
            user_text = prompt["text"]
            user_files = prompt["files"]

        # Neither of these two scenarios happened, so exit the function
        else:
            return

        # Add the current prompt to chat context
        st.session_state.messages.append({
            "role": "user",
            "content": user_text, # The text prompt
            "images": user_files.copy() # Any images sent as part of the prompt
        })

        # Display user messages and files in chat message container
        with chat_container:
            with st.chat_message("user"):
                # Display the message
                st.markdown(user_text)

                # Then display any images (if there are any)
                for file in user_files:
                    st.image(file)

                # Prepare multimodal message parts for Gemini
                # Prepare SHAP data for Gemini's context
                summary_shap_data = prepare_shap_summary_for_gemini(current_shap_1d, feature_names)

                summary_lime_data = prepare_lime_summary_for_gemini(st.session_state["last_lime_explanation"], st.session_state["last_lime_analysis_idx"])


                # Context about the figure from the "Explain this graph for me" button can be inserted here, if available
                if figure_context is not None:
                    text_part = f"""
                                            Patient Prediction: {predicted_disease} ({max(probabilities) * 100:.2f}% confidence)
                                            Key SHAP Drivers: {summary_shap_data}
                                            Clinical Threshold Context: {summary_lime_data}
                                            Graph Context: {figure_context}
                                            Explain this graph as if the user does not understand how to interpret a SHAP and/or LIME explainability graph from the diagnosis. Make it easy to understand and in simple medical terms.
                                            Use the LIME thresholds (example 'ESR <= 9.00') to explain the clinical reasoning, and SHAP to explain the relative importance.
                                        """

                else:
                    text_part = f"""
                        Patient Prediction: {predicted_disease} ({max(probabilities) * 100:.2f}% confidence)
                        Key SHAP Drivers: {summary_shap_data}
                        Clinical Threshold Context: {summary_lime_data}
                        User Question: {user_text}
                        Use the LIME thresholds (example 'ESR <= 9.00') to explain the clinical reasoning, and SHAP to explain the relative importance.
                    """

                # Flatten the text part to match Gemini's input requirements
                message_parts = [text_part]

                for file in user_files:
                    if isinstance(file, Image.Image):
                        message_parts.append(file)
                    else:
                        image = Image.open(file)
                        message_parts.append(image)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response_text = ""

                # Send the message to Gemini using the session in session_state
                response_stream = st.session_state.chat_session.send_message_stream(message=message_parts)

                # Iterate through the generator
                for chunk in response_stream:
                    # Each chunk is a GenerateContentResponse object
                    # The Gemini API provides a shortcut .text property on the chunk (instead of finding the child of many JSON objects)
                    # If a chunk is available
                    if chunk.text:
                        # Update the full response with the new chunk
                        full_response_text += chunk.text
                        # Update the UI of the response with the accumulated text and a cursor (here, a pipe)
                        response_placeholder.markdown(full_response_text + "|")

                # Final update to remove the cursor
                response_placeholder.markdown(full_response_text)

        # Save the response from Gemini to the history
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})
        st.rerun() # Refresh the chatbot to clean the UI

# App code
# Declare and set the UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("Autoimmune Rheumatic Diagnosis Decision Support System")
st.write("Please enter a patient's clinical data below to generate a diagnostic probability analysis between normal and 6 possible autoimmune rheumatic diseases.")

with st.form("clinical_form"):
    # Group 1: Demographics and inflammation
    st.subheader("1. Patient Demographics & Inflammatory Markers")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45, help="Patient age (18 to 100 years)")
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    with col2:
        esr = st.number_input("Erythrocyte Sedimentation Rate (ESR, mm/hr)", min_value=0.0, max_value=150.0, value=np.nan, help="Normal: Male < 15, Female < 20") # Previous default value: 15.0
        crp = st.number_input("C-Reactive Protein (CRP, mg/L)", min_value=0.0, max_value=300.0, value=np.nan, help="Normal range: 0.1 to 3.0 mg/L") # Previous default value: 3.0

    st.divider()

    # Group 2: Serology & specific antibodies
    st.subheader("2. Serology & Autoantibodies")
    col3, col4 = st.columns(2)

    with col3:
        rf = st.number_input("Rheumatoid Factor (RF, IU/ml)", min_value=0.0, max_value=500.0, value=np.nan, help="Normal range: 0.1 to 3.0 IU/ml") # Previous default value: 10.0
        anti_ccp = st.number_input("Anti-Cyclic Citrullinated Peptide (anti-CCP, U/mL)", min_value=0.0, max_value=500.0, value=np.nan, help="Normal range: 0.0 to 20.0 U/mL") # Previous default value: 10.0
        hla_b27 = st.selectbox("Human Leukocyte Antigen B27 (HLA-B27)", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")
        ana = st.selectbox("Anti-Nuclear Antibody (ANA)", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")

    with col4:
        anti_ro = st.selectbox("Anti-Ro/SSA Antibodies", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")
        anti_la = st.selectbox("Anti-La/SSB Antibodies", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")
        anti_dsdna = st.selectbox("Anti-double stranded DNA Antibodies (anti-dsDNA)", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")
        anti_sm = st.selectbox("Anti-Smith Antibodies (anti-Sm)", options=[np.nan, 0, 1], format_func=lambda x: "Select option..." if x is np.nan else ("Negative" if x == 0 else "Positive"), help="Choose \"Select option...\" if not tested")

    st.divider()

    # Group 3: Complement System
    st.subheader("3. Complement Components")
    col5, col6 = st.columns(2)

    with col5:
        c3 = st.number_input("Complement Component 3 (C3, mg/dL)", min_value=0.0, max_value=300.0, value=np.nan, help="Normal ranges: Male 90 to 180 mg/dL, Female 88 to 206 mg/dL") # Previous default value = 120.0

    with col6:
        c4 = st.number_input("Complement Component 4 (C4, mg/dL)", min_value=0.0, max_value=150.0, value=np.nan, help="Normal ranges: Male 12 to 72 mg/dL, Female 13 to 75 mg/dL") # Previous default value = 30.0

    st.write("")  # Spacing
    submit = st.form_submit_button("Run Autoimmune Rheumatic Diagnostic Analysis", type='primary', icon=':material/upload_file:', width='stretch')

# Processing the input once the Submit button has been clicked
# Initialise the clicked state and LLM messages if it has not existed yet
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if submit:
    st.session_state.clicked = True

    # Clear any existing LIME graph cache keys (this consideration is only done for LIME because it is computationally expensive unlike SHAP)
    for key in list(st.session_state.keys()):
        if key.startswith("lime_explainer_") or key.startswith("lime_figure_"):
            del st.session_state[key]

    with st.spinner("Analysing patient's clinical data..."):
        try:
            # 1) Gather all inputs into a dictionary matching what transform_data() expects
            # Format of transform_data()'s raw_input_data
            # ['age',
            #  'gender',
            #  'ESR',
            #  'CRP',
            #  'RF',
            #  'antiCCP',
            #  'HLA-B27',
            #  'ANA',
            #  'antiRo',
            #  'antiLa',
            #  'antiDsDNA',
            #  'antiSm',
            #  'C3',
            #  'C4']

            raw_input_data = {
                'age': age,
                'ESR': esr,
                'CRP': crp,
                'RF': rf,
                'antiCCP': anti_ccp,
                'C3': c3,
                'C4': c4,
                'gender': gender,
                'HLA-B27': hla_b27,
                'ANA': ana,
                'antiRo': anti_ro,
                'antiLa': anti_la,
                'antiDsDNA': anti_dsdna,
                'antiSm': anti_sm
            }

            # 2) Transform the raw input data
            transformed_data, _ = transform_data(raw_input_data)

            # lime_data = transform_lime_data(raw_input_data)

            # 3) Calculate the autoimmune rheumatic disease class with the highest probability
            probabilities, top_class_idx = get_prediction(transformed_data)

            # 4) Display the results of the prediction
            display_results(probabilities, top_class_idx)

            # 5) Store in session state for Gemini LLM usage later
            st.session_state.results = {
                'raw_input_data': raw_input_data,
                'probabilities': probabilities,
                'top_class_idx': top_class_idx,
                'transformed_data': transformed_data,
                # 'lime_data': lime_data
            }

        except Exception as e:
            # Extract traceback information
            exception_type, exception_object, exception_traceback = sys.exc_info()
            exception_filename = os.path.split(exception_traceback.tb_frame.f_code.co_filename)[1]
            exception_line_no = exception_traceback.tb_lineno

            # Get the full detailed string of the error
            detailed_error = traceback.format_exc()

            # Display a user-friendly message + technical details
            st.error(f"Error in {exception_filename} at Line {exception_line_no}")
            st.warning(f"Error Type: {exception_type.__name__}")
            st.info(f"Message: {e}")

            # Create an expandable section for the full Traceback (helpful for debugging)
            with st.expander("View the full error traceback"):
                st.code(detailed_error)

    display_global_interpretability(global_shap_values, global_shap_explanation, FINAL_FEATURE_NAMES, X_sample, top_class_idx)

    display_local_shap_interpretability(transformed_data, top_class_idx)

    display_local_lime_interpretability(transformed_data, top_class_idx)

    st.session_state['last_prediction'] = {
        'predicted_disease': AUTOIMMUNE_RHEUMATIC_DISEASE_CLASSES[top_class_idx],
        'probabilities': probabilities,
        'transformed_data': transformed_data,
        # 'lime_data': lime_data,
        'current_patient_shap_1d': current_patient_shap_1d
    }

    prepare_shap_summary_for_gemini(current_patient_shap_1d, FINAL_FEATURE_NAMES)

    prepare_lime_summary_for_gemini(st.session_state["last_lime_explanation"], st.session_state["last_lime_analysis_idx"])

    # run_chatbot_interface(top_disease, probabilities, FINAL_FEATURE_NAMES, current_patient_shap_1d)

# If the website is refreshed, and a button has been clicked already
elif st.session_state.clicked:
    # Retrieve the past result
    result = st.session_state.results
    last_prediction = st.session_state['last_prediction']

    # Restore the result and plots
    display_results(result['probabilities'], result['top_class_idx'])

    # Redisplay global SHAP insights
    display_global_interpretability(
        global_shap_values,
        global_shap_explanation,
        FINAL_FEATURE_NAMES,
        X_sample,
        result['top_class_idx']
    )

    # Redisplay local SHAP insights
    display_local_shap_interpretability(
        result['transformed_data'],
        result['top_class_idx']
    )

    # Redisplay local LIME insights
    display_local_lime_interpretability(
        result['transformed_data'],
        result['top_class_idx']
    )

if st.session_state.clicked:
    # Get explainer image and explainer image context if they exist, then clear them to avoid any triggers twice
    explainer_image = st.session_state.get('pending_explainer_image')
    explainer_figure_context = st.session_state.get('pending_figure_context')

    # Clear the pending state so it does not loop
    if 'pending_explainer_image' in st.session_state:
        del st.session_state['pending_explainer_image']
    if 'pending_figure_context' in st.session_state:
        del st.session_state['pending_figure_context']

    last_prediction = st.session_state['last_prediction']

    run_chatbot_interface(
        last_prediction['predicted_disease'],
        last_prediction['probabilities'],
        FINAL_FEATURE_NAMES,
        last_prediction['current_patient_shap_1d'],
        explainer_image=explainer_image,
        figure_context=explainer_figure_context
    )