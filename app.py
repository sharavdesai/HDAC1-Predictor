import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="HDAC1 Predictor", page_icon="💊", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('HDAC1_RF_Model.pkl')
        features = joblib.load('HDAC1_Feature_Names.pkl')
        return model, features
    except:
        return None, None

model, features = load_assets()

# --- HELPER FUNCTIONS ---
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        data = {'MW': mw, 'LogP': logp, 'NumHDonors': hbd, 'NumHAcceptors': hba}
        for i, bit in enumerate(fp):
            data[f'Bit_{i}'] = bit
        return data
    except:
        return None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["Single Molecule", "Batch Prediction (CSV)"])

# --- MAIN APP ---
st.title("💊 HDAC1 Inhibitor Predictor developed by MPBSCPER")

if model is None:
    st.error("⚠️ Model files not found! Please upload .pkl files to GitHub.")
    st.stop()

# MODE 1: SINGLE MOLECULE
if app_mode == "Single Molecule":
    st.subheader("1. Input Molecule")
    user_input = st.text_area("Paste SMILES string:", "ONC(=O)CCCCCCC(=O)Nc1ccccc1")
    
    if st.button("🚀 Predict Activity"):
        desc = calculate_descriptors(user_input)
        if desc:
            # Align features
            df_row = pd.DataFrame([desc])
            df_ready = pd.DataFrame(0, index=[0], columns=features)
            common = list(set(df_row.columns) & set(features))
            df_ready[common] = df_row[common]
            
            # Predict
            pIC50 = model.predict(df_ready)[0]
            ic50_nm = 10**(9 - pIC50)
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted pIC50", f"{pIC50:.2f}")
                if pIC50 > 6.0:
                    st.success("Verdict: **ACTIVE**")
                else:
                    st.error("Verdict: **INACTIVE**")
            with col2:
                st.metric("Estimated IC50", f"{ic50_nm:.1f} nM")
                mol = Chem.MolFromSmiles(user_input)
                st.image(Draw.MolToImage(mol), caption="Structure", width=250)
        else:
            st.error("Invalid SMILES.")

# MODE 2: BATCH PREDICTION
elif app_mode == "Batch Prediction (CSV)":
    st.subheader("📂 Upload CSV File")
    st.info("Upload a CSV file with a column named **'SMILES'**.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            st.write(f"Loaded {len(df)} molecules. Running predictions...")
            
            # Progress bar
            my_bar = st.progress(0)
            preds = []
            
            for i, smiles in enumerate(df['SMILES']):
                desc = calculate_descriptors(smiles)
                if desc:
                    df_row = pd.DataFrame([desc])
                    df_ready = pd.DataFrame(0, index=[0], columns=features)
                    common = list(set(df_row.columns) & set(features))
                    df_ready[common] = df_row[common]
                    pred = model.predict(df_ready)[0]
                    preds.append(pred)
                else:
                    preds.append(None)
                my_bar.progress((i + 1) / len(df))
            
            df['Predicted_pIC50'] = preds
            df['Verdict'] = df['Predicted_pIC50'].apply(lambda x: 'ACTIVE' if x > 6.0 else 'INACTIVE')
            
            st.success("✅ Screening Complete!")
            st.dataframe(df.sort_values(by='Predicted_pIC50', ascending=False))
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "HDAC1_Screening_Results.csv", "text/csv")
            
        else:
            st.error("CSV must contain a 'SMILES' column.")

st.markdown("---")
st.caption("© Sharav Desai & Shubhada Malode,2026 Pharmaceutical Biotechnology Research Lab.")
