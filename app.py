import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Pro HealthBot AI", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .auth-box { padding: 30px; border-radius: 15px; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .diag-card { padding: 20px; border-left: 8px solid #ff4b4b; background: #fff5f5; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
@st.cache_data
def load_ml_model():
    train = pd.read_csv('Training.csv')
    train.columns = [col.strip() for col in train.columns]
    docs = pd.read_csv('doctors_dataset.csv', names=['Name', 'Description'])
    le = LabelEncoder()
    X = train.iloc[:, :-1]
    y = le.fit_transform(train.iloc[:, -1])
    model = DecisionTreeClassifier(random_state=42).fit(X, y)
    return model, le, X.columns, docs

model, le, symptom_cols, doc_dataset = load_ml_model()

# --- AUTH LOGIC (Tere purane code wala logic) ---
def register_user(reg_user, reg_pass):
    if reg_user == "" or reg_pass == "":
        st.error("Username/Password empty nahi ho sakte!")
    else:
        with open(reg_user, "w") as f:
            f.write(reg_user + "\n")
            f.write(reg_pass)
        st.success("Registration Successful! Ab Login kariye.")

def login_verify(user, pwd):
    if user in os.listdir():
        with open(user, "r") as f:
            lines = f.read().splitlines()
            if len(lines) >= 2 and pwd == lines[1]:
                return True
    return False

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- UI NAVIGATION ---
if not st.session_state.logged_in:
    # --- LOGIN / SIGNUP PAGE ---
    st.title("🔐 HealthBot Secure Access")
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    
    with tab1:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Sign In"):
            if login_verify(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid Username or Password")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        new_u = st.text_input("Choose Username", key="reg_u")
        new_p = st.text_input("Choose Password", type="password", key="reg_p")
        if st.button("Register"):
            register_user(new_u, new_p)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- MAIN APP (Diagnosis) ---
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🏥 Intelligent Health Diagnostic System")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Step 1: Select Symptoms")
        display_names = [s.replace('_', ' ').title() for s in symptom_cols]
        user_selection = st.multiselect("Kya symptoms hain aapko?", options=display_names)
        
        if st.button("ANALYZE NOW"):
            if not user_selection:
                st.warning("Kam se kam ek symptom select karein.")
            else:
                # Prediction
                vec = np.zeros(len(symptom_cols))
                for s in user_selection:
                    internal_name = s.lower().replace(' ', '_')
                    if internal_name in symptom_cols:
                        vec[list(symptom_cols).index(internal_name)] = 1
                
                prediction = le.inverse_transform(model.predict([vec]))[0]
                st.session_state.result = prediction
                st.session_state.selected_symptoms = user_selection

    with col2:
        st.subheader("📊 Step 2: Diagnostic Report")
        if 'result' in st.session_state:
            st.markdown(f"""
                <div class="diag-card">
                    <h3>Probable Condition:</h3>
                    <h2 style='color:#ff4b4b;'>{st.session_state.result}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Doctor Advice
            doc_info = doc_dataset[doc_dataset['Name'].str.contains(st.session_state.result, case=False, na=False)]
            if not doc_info.empty:
                st.write(f"**Specialist to Consult:** {doc_info['Name'].values[0]}")
                st.info(f"🔗 [Get Medical Advice]({doc_info['Description'].values[0]})")
            
            st.write("**Symptoms Analyzed:**")
            st.write(", ".join(st.session_state.selected_symptoms))
        else:
            st.info("Input symptoms and click Analyze to generate report.")

st.markdown("---")
st.caption("JIIT Batch 2027 | Health Assistant Pro")