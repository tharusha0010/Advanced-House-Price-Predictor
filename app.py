import streamlit as st
import numpy as np
import joblib
import time

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
    total_sf = st.number_input('Total Square Feet', 500, 10000, 2000, step=100)
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
    
    if bedrooms > tot_rms:
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