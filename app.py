import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MediAI | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide"
)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv('Training.csv')
        train.columns = [col.strip() for col in train.columns]

        # 🔥 IMPORTANT: Shuffle dataset to remove bias
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)

        # Doctor dataset (optional)
        if os.path.exists('doctors_dataset.csv'):
            docs = pd.read_csv('doctors_dataset.csv', names=['Disease', 'Specialist', 'Description'])
        else:
            docs = pd.DataFrame(columns=['Disease', 'Specialist', 'Description'])

        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 🔥 BETTER MODEL
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        return model, le, X.columns, docs, train

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


model, le, symptom_cols, doc_dataset, train_data = load_data()

# ==================== SYMPTOM MAPPING ====================
symptom_map = {
    col.replace('_', ' ').title(): col
    for col in symptom_cols
}

# ==================== HEADER ====================
st.title("⚕️ MediAI - Clinical Decision Support System")

# ==================== UI ====================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Patient Assessment")

    selected_display = st.multiselect(
        "Select Symptoms",
        options=list(symptom_map.keys())
    )

    analyze = st.button("Generate Diagnosis")

with col2:
    if analyze:
        if not selected_display:
            st.warning("Please select at least one symptom.")
        else:
            try:
                # ==================== INPUT VECTOR ====================
                input_vector = np.zeros(len(symptom_cols))

                for s in selected_display:
                    if s in symptom_map:
                        actual_col = symptom_map[s]
                        idx = list(symptom_cols).index(actual_col)
                        input_vector[idx] = 1

                # 🔍 DEBUG (optional)
                # st.write("Active symptoms:", int(np.sum(input_vector)))

                # ==================== FIX: Use DataFrame ====================
                input_df = pd.DataFrame([input_vector], columns=symptom_cols)

                # ==================== PREDICTION ====================
                prediction = model.predict(input_df)[0]
                disease = le.inverse_transform([prediction])[0]

                # ==================== PROBABILITY (TOP 3) ====================
                probs = model.predict_proba(input_df)[0]
                top_indices = np.argsort(probs)[-3:][::-1]

                st.success(f"🩺 Predicted Disease: {disease}")

                st.subheader("Top Possible Conditions")
                for idx in top_indices:
                    st.write(f"{le.inverse_transform([idx])[0]} → {probs[idx]*100:.2f}%")

                # ==================== CONFIDENCE ====================
                disease_data = train_data[train_data.iloc[:, -1] == disease]

                if not disease_data.empty:
                    expected = sum(disease_data.iloc[:, :-1].iloc[0] > 0)
                    confidence = min(100, (len(selected_display) / expected * 100)) if expected > 0 else 75
                else:
                    confidence = 75
                    expected = 0

                st.info(f"Confidence: {confidence:.2f}%")
                st.write(f"Evidence Match: {len(selected_display)}/{expected}")

                # ==================== DOCTOR RECOMMENDATION ====================
                st.subheader("Recommended Specialist")

                match = doc_dataset[doc_dataset.iloc[:, 0].str.strip() == disease]

                if not match.empty:
                    for _, row in match.iterrows():
                        st.write(f"👨‍⚕️ {row.iloc[1]}")
                        st.caption(row.iloc[2])
                else:
                    referral = "General Physician"

                    if "skin" in disease.lower():
                        referral = "Dermatologist"
                    elif "heart" in disease.lower():
                        referral = "Cardiologist"
                    elif "diabetes" in disease.lower():
                        referral = "Endocrinologist"

                    st.write(f"🏥 Consult: {referral}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:
        st.info("Select symptoms and click 'Generate Diagnosis'")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("⚠️ Disclaimer: This system is for educational purposes only.")