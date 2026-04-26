import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MediAI | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PROFESSIONAL UI STYLING ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --primary-blue: #0A58CA;
        --soft-blue: #F0F7FF;
        --slate-900: #0F172A;
        --slate-600: #475569;
        --slate-200: #E2E8F0;
        --success-green: #10B981;
    }

    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 2rem 2rem;
        border-bottom: 1px solid var(--slate-200);
        margin: -6rem -5rem 2rem -5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 999;
        margin: 0rem -5rem 2rem -5rem;
    }

    .brand-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--slate-900);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .status-badge {
        background: var(--soft-blue);
        color: var(--primary-blue);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }

    .diagnosis-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--slate-200);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    .metric-box {
        text-align: center;
        padding: 1rem;
        background: var(--soft-blue);
        border-radius: 12px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-blue);
    }

    .stButton>button {
        width: 100%;
        background-color: var(--slate-900) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .doctor-card {
        background: #ffffff;
        border-left: 4px solid var(--primary-blue);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    .doctor-name {
        color: var(--slate-900);
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
    }

    .doctor-desc {
        color: var(--slate-600);
        font-size: 0.9rem;
    }

    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid var(--slate-200);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA & LOGIC ====================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv('Training.csv')
        train.columns = [col.strip() for col in train.columns]
        
        # Mapping diseases to specialists based on your doctors_dataset.csv structure
        # (Assuming Name = Specialist Title/Dept, Description = Details/URL)
        if os.path.exists('doctors_dataset.csv'):
            docs = pd.read_csv('doctors_dataset.csv', names=['Disease', 'Specialist', 'Description'])
        else:
            # Emergency fallback structure
            docs = pd.DataFrame(columns=['Disease', 'Specialist', 'Description'])
        
        le = LabelEncoder()
        X = train.iloc[:, :-1]
        y = le.fit_transform(train.iloc[:, -1])
        
        model = DecisionTreeClassifier(max_depth=12, random_state=42)
        model.fit(X, y)
        return model, le, X.columns, docs, train
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

model, le, symptom_cols, doc_dataset, train_data = load_data()

# ==================== HEADER ====================
st.markdown(f"""
    <div class="main-header">
        <div class="brand-text">
            <span style="color: var(--primary-blue)">✚</span> MediAI Professional
        </div>
        <div class="status-badge">
            Clinical Decision Support System
        </div>
    </div>
""", unsafe_allow_html=True)

# ==================== MAIN INTERFACE ====================
col1, space, col2 = st.columns([1, 0.1, 1.2])

with col1:
    st.markdown("### Patient Assessment")
    st.markdown("Select all observable clinical symptoms to begin analysis.")
    
    clean_symptoms = [s.replace('_', ' ').title() for s in symptom_cols]
    selected_display = st.multiselect(
        "Reported Symptoms",
        options=clean_symptoms,
        placeholder="Search symptoms...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_trigger = st.button("Generate Diagnostic Report")
    
    with st.expander("System Intelligence"):
        st.info("Uses a Decision Tree Model to cross-reference reported symptoms against clinical patterns.")

with col2:
    if analyze_trigger:
        if not selected_display:
            st.warning("Please select at least one clinical symptom.")
        else:
            # Prediction Logic
            input_vector = np.zeros(len(symptom_cols))
            for s in selected_display:
                idx = list(symptom_cols).index(s.lower().replace(' ', '_'))
                input_vector[idx] = 1
            
            prediction = model.predict([input_vector])[0]
            disease = le.inverse_transform([prediction])[0]
            
            # Confidence Calculation
            disease_data = train_data[train_data.iloc[:, -1] == disease]
            expected = sum(disease_data.iloc[:, :-1].iloc[0] > 0)
            confidence = min(100, (len(selected_display) / expected * 100)) if expected > 0 else 75

            # Output UI
            st.markdown(f"""
                <div class="diagnosis-card">
                    <div class="result-label">Preliminary Diagnostic Analysis</div>
                    <div class="result-value">{disease}</div>
                    <hr>
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1" class="metric-box">
                            <div style="font-size: 0.8rem; color: var(--slate-600)">Confidence Score</div>
                            <div class="metric-value">{confidence:.1f}%</div>
                        </div>
                        <div style="flex: 1" class="metric-box">
                            <div style="font-size: 0.8rem; color: var(--slate-600)">Evidence Match</div>
                            <div class="metric-value">{len(selected_display)}/{expected}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Referral Section with Specialist from doctors_dataset.csv
            st.markdown("### Professional Referral")
            
            # Filter logic: We look for the predicted disease in our doctor dataset
            # We assume your CSV has a column where the disease name is stored
            # (Matches 'disease' column or similar)
            match = doc_dataset[doc_dataset.iloc[:, 0].str.strip() == disease]

            if not match.empty:
                for idx, row in match.iterrows():
                    st.markdown(f"""
                        <div class="doctor-card">
                            <div class="doctor-name">👨‍⚕️ Specialist: {row.iloc[1]}</div>
                            <div class="doctor-desc">{row.iloc[2]}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Standard Department Fallback logic if CSV match fails
                referral = "Internal Medicine"
                if "Skin" in disease or "Fungal" in disease: referral = "Dermatology"
                elif "Heart" in disease or "Hypertension" in disease: referral = "Cardiology"
                elif "Diabetes" in disease: referral = "Endocrinology"
                
                st.markdown(f"""
                    <div class="doctor-card">
                        <div class="doctor-name">🏥 Recommended Department: {referral}</div>
                        <div class="doctor-desc">Please consult a general practitioner or {referral.lower()} specialist for a definitive clinical evaluation.</div>
                    </div>
                """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
            <div style="height: 300px; display: flex; align-items: center; justify-content: center; 
                        border: 2px dashed var(--slate-200); border-radius: 16px; color: var(--slate-600);">
                Waiting for clinical input...
            </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
    <div style="margin-top: 5rem; padding: 2rem; text-align: center; border-top: 1px solid var(--slate-200);">
        <p style="color: var(--slate-600); font-size: 0.85rem;">
            <b>Medical Disclaimer:</b> Educational use only. Consult healthcare professionals for medical advice.
        </p>
    </div>
""", unsafe_allow_html=True)