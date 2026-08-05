import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Advanced House Price Predictor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('notebooks/web_model.pkl')
    except FileNotFoundError:
        return joblib.load('web_model.pkl')

model = load_ml_model()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1018/1018671.png", width=120)
    st.title("About the Project")
    st.info(
        "This Machine Learning Web App leverages an XGBoost Regression model "
        "trained on the Ames Housing Dataset. It analyzes key architectural "
        "and structural features to estimate the real estate market value."
    )
    st.markdown("---")
    st.markdown("🚀 **Version:** 1.0.0")
    st.markdown("🧠 **Core AI Model:** XGBoost")

st.title('🏘️ Advanced House Price Predictor')
st.markdown("Provide the specifications of the property below to instantly generate an AI-driven market value estimation.")
st.markdown('---')

st.subheader("📋 Property Specifications")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📐 Structure & Quality")
    overall_qual = st.slider('Overall Quality (1-10)', 1, 10, 5, help="1 = Very Poor, 10 = Very Excellent")
    total_sf = st.number_input('Total Square Feet', value=2000, step=100)
    house_age = st.number_input('House Age (Years)', 0, 150, 10, step=1)

with col2:
    st.markdown("#### 🛏️ Rooms & Layout")
    tot_rms = st.slider('Total Rooms (Above Ground)', 2, 15, 6)
    bedrooms = st.slider('Total Bedrooms', 0, 10, 3)
    full_bath = st.slider('Full Bathrooms', 1, 5, 2)

with col3:
    st.markdown("#### 🚘 Amenities")
    garage_cars = st.slider('Garage Capacity (Cars)', 0, 5, 2)
    fireplaces = st.slider('Fireplaces', 0, 4, 0)
    central_air = st.selectbox('Central Air Conditioning', ['Yes', 'No'])

central_air_val = 1 if central_air == 'Yes' else 0

st.markdown('---')

col_btn, col_result = st.columns([1, 2])

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button('🎯 Predict Price', use_container_width=True, type="primary")

if predict_button:
    is_valid = True
    
    # Custom Validation for Minimum Square Feet
    if total_sf < 100:
        st.error("❌ Validation Error: Total square feet must be at least 100. Please enter a valid property size.")
        is_valid = False
        
    elif bedrooms > tot_rms:
        st.error("❌ Validation Error: Total bedrooms cannot exceed the total number of rooms.")
        is_valid = False
        
    elif (total_sf / tot_rms) < 120:
        st.warning("⚠️ Smart Alert: The number of rooms seems unusually high for the given square footage. The prediction may be inaccurate.")
        
    if is_valid:
        with st.spinner('Analyzing property features...'):
            time.sleep(1)
            
            input_data = np.array([[
                overall_qual, total_sf, garage_cars, central_air_val, 
                full_bath, fireplaces, house_age, tot_rms, bedrooms
            ]])
            
            predicted_log_price = model.predict(input_data)[0]
            actual_price = np.exp(predicted_log_price)
            
        with col_result:
            st.metric(label="Estimated Market Value", value=f"${actual_price:,.2f}")
            
            report_data = {
                "Overall Quality (1-10)": [overall_qual],
                "Total Square Feet": [total_sf],
                "House Age (Years)": [house_age],
                "Total Rooms": [tot_rms],
                "Total Bedrooms": [bedrooms],
                "Full Bathrooms": [full_bath],
                "Garage Capacity (Cars)": [garage_cars],
                "Fireplaces": [fireplaces],
                "Central Air": [central_air],
                "Estimated Market Value ($)": [round(actual_price, 2)]
            }
            
            report_df = pd.DataFrame(report_data)
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📄 Download Property Report (CSV)",
                data=csv_data,
                file_name="Property_Valuation_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

        # --- EXPLAINABLE AI (SHAP) SECTION ---
        st.markdown("---")
        st.subheader("🧠 Explainable AI: Price Drivers")
        st.info("The chart below explains how each feature influenced the final predicted price. Red bars push the price higher, while blue bars pull it lower.")
        
        with st.spinner('Generating AI explanation...'):
            feature_names = [
                'Overall Quality', 'Total Sq Ft', 'Garage Cars', 'Central Air', 
                'Full Bathrooms', 'Fireplaces', 'House Age', 'Total Rooms', 'Total Bedrooms'
            ]
            
            input_df = pd.DataFrame(input_data, columns=feature_names)
            
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            
            st.pyplot(fig, clear_figure=True)