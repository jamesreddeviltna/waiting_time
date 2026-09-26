%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model and preprocessors ---
@st.cache_resource
def load_model_and_preprocessors():
    rf_model = joblib.load('random_forest_regressor_model.joblib')
    encoder = joblib.load('one_hot_encoder.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return rf_model, encoder, feature_columns

rf_model, encoder, feature_columns = load_model_and_preprocessors()

# --- Define dropdown options (explicitly from the notebook's df_filtered) ---
radiology_names = ['Abdomen Supine & Upright', 'Abdomen Upright', 'Acute Abdomen series (Chest,Abdomen Supine,Abdomen Upright)', 'Ankle Lt.AP,Lateral', 'Ankle Lt.AP,Lateral,Mortise', 'Ankle Rt. AP,Lateral', 'Ankle Rt. AP,Lateral,Mortise', 'Big Toe Rt. AP,Lateral', 'CT Abdomen (Whole) (With Contrast)', 'CT Abdomen (Whole) (With Contrast)(3 Phase)', 'CT Abdomen(Lower)(With Contrast)', 'CT Brain (Without Contrast)', 'CT Brain + C spine (Without Contrast)', 'CT Brain + facial bone (Without Contrast)', 'CT Brain (With Contrast)', 'CT Brain (With & Without Contrast)', 'CT Cervical Spine (Without Contrast)', 'CT Chest (High Resolution)', 'CT Chest (Low Dose)', 'CT Chest (Without Contrast)', 'CT Chest (With & Without Contrast)', 'CT Lumbar Spine (Without Contrast)', 'CT Orbit (Without Contrast)', 'CT Para-nasal Sinus (Without Contrast)', 'CT Pelvis (Without Contrast)', 'CT Whole Spine (Without Contrast)', 'Cervical Spine AP,Lateral', 'Cervical Spine AP,Lateral,Oblique', 'Chest PA Upright', 'Chest PA Upright (Mobile)', 'Chest PA Upright (Portable)', 'Chest X-Ray (mobile CH- up)', 'Clavicle AP', 'Coccyx AP,Lateral', 'Elbow Both AP,Lateral', 'Elbow Both AP,Lateral,Oblique', 'Elbow Lt. AP,Lateral', 'Elbow Lt. AP,Lateral,Oblique', 'Elbow Rt. AP,Lateral', 'Elbow Rt. AP,Lateral,Oblique', 'Foot Lt. AP,Oblique', 'Foot Lt. AP,Oblique,Lateral', 'Foot Rt. AP,Oblique', 'Foot Rt. AP,Oblique,Lateral', 'Forearm Both AP,Lateral', 'Forearm Lt. AP,Lateral', 'Forearm Rt. AP,Lateral', 'Hand Lt. PA,Oblique', 'Hand Lt. PA,Oblique,Lateral', 'Hand Rt. PA,Oblique', 'Hand Rt. PA,Oblique,Lateral', 'Hip Joint Lt. AP', 'Hip Joint Rt. AP', 'Humerus Both AP,Lateral', 'Humerus Lt. AP,Lateral', 'Humerus Rt. AP,Lateral', 'Knee Both AP,Lateral', 'Knee Lt. AP,Lateral', 'Knee Lt. AP,Lateral,Oblique', 'Knee Rt. AP,Lateral', 'Knee Rt. AP,Lateral,Oblique', 'Lumbo-sacral Spine AP,Lateral', 'Lumbo-sacral Spine AP,Lateral,Oblique', 'Lumbo-sacral Spine Lat (Flexion-Extension)', 'Mandible AP,Lateral,PA', 'Nasal Bone Lateral', 'Orbit Rt. AP,Lateral', 'Pelvis AP', 'Ribs Both PA,Oblique', 'Ribs Lt. PA,Oblique', 'Ribs Rt. PA,Oblique', 'Sacrum AP,Lateral', 'Shoulder Lt. AP', 'Shoulder Lt. AP,Lateral (Internal-External)', 'Shoulder Lt. AP,Lateral (Internal-External),Axial', 'Shoulder Lt. AP,Lateral (scapular-Y)', 'Shoulder Lt. AP,Lateral (transcapular)', 'Shoulder Rt. AP', 'Shoulder Rt. AP,Lateral (Internal-External)', 'Shoulder Rt. AP,Lateral (Internal-External),Axial', 'Shoulder Rt. AP,Lateral (scapular-Y)', 'Shoulder Rt. AP,Lateral (transcapular)', 'Skull AP,Lateral View', 'Soft tissue Neck AP,Lateral', 'Soft tissue Neck Lateral', 'Tibia-Fibula Lt. AP,Lateral', 'Tibia-Fibula Rt. AP,Lateral', 'Thoracic Spine AP,Lateral', 'Thoraco Lumbar Spine AP,Lateral', 'Wrist Both PA,Lateral', 'Wrist Lt. PA,Lateral', 'Wrist Lt. PA,Lateral,Oblique', 'Wrist Rt. PA,Lateral', 'Wrist Rt. PA,Lateral,Oblique']
facilities = ['ห้อง CT SCAN', 'ห้อง GENERAL X-RAY', 'ห้อง MAMMOGRAM', 'ห้อง ULTRASOUND']
sbus = ['Check up(CHK)', 'ICU', 'Mobile check up(MCHK)', 'NS', 'OR-2', 'PACU', 'Regenerative medicine + Stem cells(REG)', 'Ward5', 'Ward6', 'กุมารเวชกรรม(PED)', 'คลินิกทั่วไป(GP)', 'ความงามและศัลยกรรมตกแต่ง(AES)', 'ศัลยกรรม(SUR)', 'ศัลยกรรมกระดูกและข้อ(ORT)', 'ศูนย์หัวใจและหลอดเลือด(CDO)', 'สูตินรีเวชกรรม(OBG)', 'อายุรกรรม(MED)']
order_days_str = ['0', '1', '2', '3', '4', '5', '6']
order_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
order_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

# --- Streamlit UI ---
st.title('Duration (Min) Prediction App')
st.write('Enter the details to predict the duration of radiology procedures.')

# Input widgets
selected_radiology_name = st.selectbox('Radiology Name', radiology_names)
selected_facility = st.selectbox('Facility', facilities)
selected_sbu = st.selectbox('SBU', sbus)
selected_order_day = st.selectbox('Order Day (0=Monday, 6=Sunday)', order_days_str)
selected_order_hour = st.selectbox('Order Hour', order_hours)
selected_order_month = st.selectbox('Order Month', order_months)

# Prediction button
if st.button('Predict Duration'):
    # Prepare input for prediction
    ohe_cols = ["Radiology Name", "Facility", "SBU"]

    # Create a temporary DataFrame for categorical features for encoding
    temp_cat_df = pd.DataFrame([[selected_radiology_name, selected_facility, selected_sbu]], columns=ohe_cols)

    # Use the loaded encoder to transform these categorical features
    encoded_cat = encoder.transform(temp_cat_df)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(ohe_cols))

    # Prepare the rest of the features
    input_data_dict = {
        'Order_Day': str(selected_order_day), # Ensure it's string as in X
        'Hour_sin': np.sin(2 * np.pi * selected_order_hour / 24),
        'Hour_cos': np.cos(2 * np.pi * selected_order_hour / 24),
        'Month_sin': np.sin(2 * np.pi * (selected_order_month - 1) / 12),
        'Month_cos': np.cos(2 * np.pi * (selected_order_month - 1) / 12)
    }
    
    # Create a base DataFrame with all expected columns and fill with 0
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Populate the numerical/derived columns
    for col, value in input_data_dict.items():
        if col in input_df.columns:
            input_df[col] = value

    # Populate the one-hot encoded columns
    for col in encoded_cat_df.columns:
        if col in input_df.columns: # Only add if the column exists in the training data's features
            input_df[col] = encoded_cat_df[col].iloc[0] # Take the single row value

    # Ensure the column order matches the training data
    input_df = input_df[feature_columns]

    # Make prediction
    prediction = rf_model.predict(input_df)[0]

    st.subheader(f'Predicted Duration (Min): {prediction:.2f}')