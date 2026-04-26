import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Health Assistant", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        border-radius: 20px; 
        background: linear-gradient(45deg, #ff4b4b, #ff7676);
        color: white; font-weight: bold; border: none;
    }
    .report-card { 
        padding: 30px; border-radius: 15px; 
        background-color: white; border-left: 10px solid #ff4b4b;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    train = pd.read_csv('Training.csv')
    train.columns = [col.strip() for col in train.columns]
    docs = pd.read_csv('doctors_dataset.csv', names=['Name', 'Description'])
    le = LabelEncoder()
    X = train.iloc[:, :-1]
    y = le.fit_transform(train.iloc[:, -1])
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, le, X.columns, docs

model, le, symptom_cols, doc_dataset = load_data()

st.title("🏥 Smart Health AI")
st.write("Advance Diagnostic System | JIIT Batch of 2027")

# --- UI LAYOUT ---
col_input, col_display = st.columns([1, 1])

with col_input:
    st.subheader("🔍 Symptom Input")
    display_names = [s.replace('_', ' ').title() for s in symptom_cols]
    selected_display = st.multiselect("Select your symptoms:", options=display_names)
    
    run_btn = st.button("RUN ANALYSIS")

with col_display:
    if run_btn and selected_display:
        # Vector prep
        input_vector = np.zeros(len(symptom_cols))
        for s_disp in selected_display:
            s_internal = s_disp.lower().replace(' ', '_')
            if s_internal in symptom_cols:
                input_vector[list(symptom_cols).index(s_internal)] = 1
        
        # Predict
        disease = le.inverse_transform(model.predict([input_vector]))[0]
        
        # Display Result
        st.markdown(f"""
            <div class="report-card">
                <h3 style='margin:0;'>Preliminary Diagnosis</h3>
                <h1 style='color: #ff4b4b; margin:0;'>{disease}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Specialist Lookup
        doc_row = doc_dataset[doc_dataset['Name'].str.contains(disease, case=False, na=False)]
        if not doc_row.empty:
            st.warning(f"**Recommended Specialist:** {doc_row['Name'].values[0]}")
            st.info(f"🔗 [Medical Resource]({doc_row['Description'].values[0]})")
        
    elif run_btn:
        st.error("Please select at least one symptom.")
    else:
        st.info("👈 Select symptoms and click 'Run Analysis' to see results.")

st.markdown("---")
st.caption("Developed by  | Academic Project | Not for medical emergency use.")