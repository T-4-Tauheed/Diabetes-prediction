import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from io import BytesIO

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# ──────────────── CUSTOM NEON THEME ────────────────
PRO_THEME = """
<style>
body, .stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    color: #e0e0e0 !important;
}
h1 { color: #00ffcc !important; text-shadow: 0 0 8px #00ffcc; }
h2 { color: #ffcc00 !important; text-shadow: 0 0 8px #ffcc00; }
h3 { color: #1abc9c !important; text-shadow: 0 0 8px #1abc9c; }
h4, h5 { color: #ff9933 !important; text-shadow: 0 0 6px #ff9933; }

/* Inputs */
.stNumberInput label, .stTextInput label { 
    color: #ffcc00 !important; 
    font-weight: 600; 
}
.stNumberInput input, .stTextInput input {
    color: #000000 !important; 
    font-weight: bold;
    background-color: #f5f5f5 !important;
    border-radius: 6px;
}
.stTextInput {
    max-width: 300px !important;
}

/* Predict Button Styling */
.stForm button {
    background: linear-gradient(90deg, #00ffcc, #0099ff) !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 0.5em 1.5em !important;
    border: none !important;
}
</style>
"""
st.markdown(PRO_THEME, unsafe_allow_html=True)

# ──────────────── CUSTOM TAB THEME ────────────────
TAB_STYLE = """
<style>
/* Streamlit Tabs Styling */
.stTabs [data-baseweb="tab"] {
    background-color: #1f2c34 !important;   /* Dark gray tab background */
    color: #ffffff !important;              /* White text */
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease-in-out;
    border: 1px solid #00ffcc33 !important; /* subtle border */
}
.stTabs [data-baseweb="tab"]:hover {
    box-shadow: 0 0 10px #00ffcc;           /* glow on hover */
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #00ffcc, #0099ff) !important; /* Neon gradient when active */
    color: black !important;
    box-shadow: 0 0 12px #00ffcc;           /* glowing active tab */
}
</style>
"""
st.markdown(TAB_STYLE, unsafe_allow_html=True)

# ──────────────── EXTRA STYLES (your requests) ────────────────
EXTRA_STYLE = """
<style>
/* Sidebar headings: Project By & Adjust Patient Values */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: #00ffcc !important;       /* neon cyan */
    text-shadow: 0 0 6px #00ffcc;    /* glow */
    font-weight: bold !important;
}

/* Download button styling */
.stDownloadButton button {
    background: linear-gradient(90deg, #00ffcc, #0099ff) !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 0.6em 1.5em !important;
    border: none !important;
    box-shadow: 0 0 8px #00ffcc;
    transition: all 0.3s ease-in-out;
}
.stDownloadButton button:hover {
    box-shadow: 0 0 16px #00ffcc, 0 0 20px #0099ff;
    transform: scale(1.03);
}

/* Custom result badges */
.neon-result {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: bold;
    font-size: 16px;
    color: black;
    background: linear-gradient(90deg, #00ffcc, #0099ff);
    box-shadow: 0 0 12px #00ffcc;
}
.neon-result.bad {
    background: linear-gradient(90deg, #ff0066, #ff3333);
    box-shadow: 0 0 12px #ff3366;
    color: white;
}
</style>
"""
st.markdown(EXTRA_STYLE, unsafe_allow_html=True)

# ───────────────────────── FEATURES ─────────────────────────
FEATURES = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
LIMITS = {"Glucose": (0, 199), "BloodPressure": (0, 122), "Insulin": (0, 846), "BMI": (0.0, 67.1), "Age": (21, 81)}
DEFAULTS = {"Glucose": 117, "BloodPressure": 72, "Insulin": 30, "BMI": 32.0, "Age": 29}

# ───────────────────────── LOAD MODEL ─────────────────────────
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model("diabetes_model.pkl")
except Exception as e:
    st.error(f"❌ Could not load model. Error: {e}")
    st.stop()

# ───────────────────────── HELPERS ─────────────────────────
def predict_one(row_1x5: np.ndarray):
    pred = int(model.predict(row_1x5)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(row_1x5)[0][1])
        except Exception:
            proba = None
    return pred, proba

def risk_label(prob):
    if prob is None: return "Unknown", "#bdc3c7"
    if prob < 0.34: return "Low", "#2ecc71"
    if prob < 0.67: return "Medium", "#f1c40f"
    return "High", "#e74c3c"

def generate_text_report(name, glucose, bp, insulin, bmi, age, result, risk, prob):
    """Generates a simple, valid text report as a byte stream."""
    lines = []
    lines.append("===== Diabetes Prediction Report =====")
    lines.append(f"Name: {name or 'N/A'}")
    lines.append(f"Glucose: {glucose}")
    lines.append(f"Blood Pressure: {bp}")
    lines.append(f"Insulin: {insulin}")
    lines.append(f"BMI: {bmi}")
    lines.append(f"Age: {age}")
    lines.append("")
    lines.append(f"Prediction: {result}")
    lines.append(f"Risk Level: {risk}")
    lines.append(f"Probability: {prob}")
    lines.append("")
    if result == "Diabetic":
        lines.append("Consult with doctor")
        lines.append("Contact diabetes specialist: https://www.google.com/search?q=diabetes+doctor+near+me")
    else:
        lines.append("Tip: Maintain a balanced diet and regular exercise routine.")
    lines.append("")
    lines.append("⚠️ Educational use only — not medical advice.")
    lines.append("")
    lines.append("──────────────────────────────────────")
    lines.append("Prepared by: Tauheed Akhtar (UON)")
    lines.append("Project: Diabetes Prediction App")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    text_content = "\n".join(lines)
    return text_content.encode('utf-8')

# ───────────────────────── SIDEBAR ─────────────────────────
st.sidebar.title("👤 Project By")
st.sidebar.success(
    "**Name:** Tauheed Akhtar\n"
    "**University:** UON\n"
    "**Project:** Diabetes Prediction App"
)

st.sidebar.markdown("### ⚙️ Adjust Patient Values")
st.session_state.glucose = st.sidebar.slider("Glucose (mg/dL)", *LIMITS["Glucose"], DEFAULTS["Glucose"])
st.session_state.bp = st.sidebar.slider("Blood Pressure (mm Hg)", *LIMITS["BloodPressure"], DEFAULTS["BloodPressure"])
st.session_state.insulin = st.sidebar.slider("Insulin (mu U/ml)", *LIMITS["Insulin"], DEFAULTS["Insulin"])
st.session_state.bmi = st.sidebar.slider("BMI (kg/m²)", float(LIMITS["BMI"][0]), float(LIMITS["BMI"][1]), DEFAULTS["BMI"], step=0.1)
st.session_state.age = st.sidebar.slider("Age (years)", LIMITS["Age"][0], 120, DEFAULTS["Age"])

# ───────────────────────── MAIN ─────────────────────────
tab1, tab2 = st.tabs(["🏥 Diabetes Prediction", "🗺️ Nearby Hospitals"])

# ====================== TAB 1: DIABETES PREDICTION ======================
with tab1:
    st.title("🩺 Diabetes Prediction App")
    st.subheader("📝 Enter Patient Details")

    with st.form("single_form"):
        # Centered inputs with minimal width
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            name = st.text_input("Patient Name", "")

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            glucose = st.number_input("Glucose (mg/dL)", *LIMITS["Glucose"], value=st.session_state.glucose)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", *LIMITS["BloodPressure"], value=st.session_state.bp)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            insulin = st.number_input("Insulin (mu U/ml)", *LIMITS["Insulin"], value=st.session_state.insulin)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            bmi = st.number_input("BMI (kg/m²)", float(LIMITS["BMI"][0]), float(LIMITS["BMI"][1]), 
                                value=float(st.session_state.bmi), step=0.1)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            age = st.number_input("Age (years)", LIMITS["Age"][0], 120, value=st.session_state.age)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        row = np.array([[glucose, blood_pressure, insulin, bmi, age]], dtype=float)
        pred, proba = predict_one(row)

        healthy = (70 <= glucose <= 120 and 60 <= blood_pressure <= 80 and
                15 <= insulin <= 276 and 18.5 <= bmi <= 24.9 and 18 <= age <= 60)

        if healthy:
            st.markdown(f"<div class='neon-result'>✅ {name or 'Patient'} has not diabetes</div>", unsafe_allow_html=True)
            pred = 0
        elif pred == 1:
            st.markdown(f"<div class='neon-result bad'>⚠️ {name or 'Patient'} has diabetes</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='neon-result'>✅ {name or 'Patient'} has not diabetes</div>", unsafe_allow_html=True)

        # Risk info
        label, rcolor = risk_label(proba)
        pct_text = "N/A" if proba is None else f"{proba*100:.2f}%"
        if proba is not None:
            st.progress(int(proba * 100))

        result = "Diabetic" if pred == 1 else "Not Diabetic"

        # Result card
        card_color = "#00ccff" if pred == 1 else "#00ff99"
        st.markdown(
            f"""
            <div style="border-left:10px solid {card_color}; background:rgba(0,0,0,0.6); padding:16px 18px; border-radius:12px; margin-bottom:10px;">
            <div style="font-weight:700; color:{card_color}; font-size:18px; margin-bottom:6px;">Prediction Result</div>
            <div style="color:#ddd; line-height:1.5;">
                <b>Name:</b> {name or 'N/A'}<br/>
                <b>Estimated Risk:</b> {pct_text}  
                <span style='background:{card_color}; color:black; padding:2px 8px; border-radius:999px;'> {label} </span><br/>
                <b>Inputs:</b> Glucose={glucose}, BP={blood_pressure}, Insulin={insulin}, BMI={bmi}, Age={age}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Suggestions
        if pred == 1:
            st.subheader("👨‍⚕️ Consult with Doctor")
            st.markdown("[Contact diabetes specialist](https://www.google.com/search?q=diabetes+doctor+near+me)", unsafe_allow_html=True)

            st.subheader("🥗 Diet and Exercise Plan for Diabetics")
            st.markdown("""
            **Dietary Suggestions:**
            * **Focus on Low-Glycemic Foods:** Choose complex carbohydrates like whole grains, vegetables, and legumes.
            * **Lean Proteins:** Include chicken, fish, beans, and tofu in your meals.
            * **Healthy Fats:** Incorporate avocados, nuts, and olive oil to help manage blood sugar levels.
            * **Limit Sugary Drinks:** Avoid sodas, juices, and other sweetened beverages.
            * **Portion Control:** Be mindful of portion sizes to manage calorie intake.
            
            **Exercise Suggestions:**
            * **Aerobic Exercise:** Aim for at least 150 minutes of moderate-intensity activities per week, such as brisk walking, cycling, or swimming.
            * **Strength Training:** Incorporate activities like weight lifting or bodyweight exercises at least twice a week to build muscle mass, which helps improve insulin sensitivity.
            * **Consistency is Key:** Regular physical activity helps your body use insulin more effectively and can lower blood sugar levels.
            """)
        else:
            st.info("✅ Maintain a healthy lifestyle with balanced diet and regular exercise.")

        # Download Report
        txt_bytes = generate_text_report(name, glucose, blood_pressure, insulin, bmi, age, result, label, pct_text)
        st.download_button("⬇️ Download Report (TXT)", data=txt_bytes,
                        file_name=f"{name or 'patient'}_report.txt", mime="text/plain")

# ====================== TAB 2: HOSPITAL MAP ======================
with tab2:
    st.title("🗺️ Nearby Hospitals")
    st.write("Here are some hospitals near you:")

    # Example hospital coordinates (Lahore — replace with your city if needed)
    hospital_data = pd.DataFrame({
        'lat': [31.5204, 31.5091, 31.5330],
        'lon': [74.3587, 74.3306, 74.3667],
        'name': ["Shaukat Khanum Hospital", "Mayo Hospital", "Jinnah Hospital"]
    })

    # Show interactive map
    st.map(hospital_data)

    # List hospital names
    st.subheader("🏥 Hospital List")
    for i, row in hospital_data.iterrows():
        st.write(f"🔹 {row['name']} (Lat: {row['lat']}, Lon: {row['lon']})")
