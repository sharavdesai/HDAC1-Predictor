from rdkit.Chem import Draw
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HDAC1 Predictor-MPB-SCPER", page_icon="💊", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load model and feature names
    try:
        model = joblib.load('HDAC1_RF_Model.pkl')
        features = joblib.load('HDAC1_Feature_Names.pkl')
        return model, features
    except Exception as e:
        return None, None

model, features = load_assets()

# --- HELPER FUNCTIONS ---
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate the 4 Lipinski Props
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Calculate the 1024 Bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    
    # Combine
    data = {'MW': mw, 'LogP': logp, 'NumHDonors': hbd, 'NumHAcceptors': hba}
    for i, bit in enumerate(fp):
        data[f'Bit_{i}'] = bit
    return data

# --- GUI LAYOUT ---
st.title("💊 HDAC1 Inhibitor Predictor-MPB-SCPER")
st.markdown("### Research-Grade Screening Tool")
st.info("Developed using Random Forest (R²=0.66) on ChEMBL Data.")

# Check if model loaded
if model is None:
    st.error("⚠️ Model files not found! Please ensure .pkl files are in the directory.")
else:
    # INPUT SECTION
    st.subheader("1. Input Molecule")
    user_input = st.text_area("Paste SMILES string:", "ONC(=O)CCCCCCC(=O)Nc1ccccc1", help="Enter a valid SMILES string.")

    if st.button("🚀 Predict Activity"):
        if not user_input:
            st.warning("Please enter a SMILES string.")
        else:
            with st.spinner("Analyzing chemical structure..."):
                # 1. Calc Descriptors
                desc = calculate_descriptors(user_input)
                
                if desc:
                    # 2. Align features
                    df_input = pd.DataFrame([desc])
                    df_ready = pd.DataFrame(0, index=[0], columns=features)
                    common_cols = list(set(df_input.columns) & set(features))
                    df_ready[common_cols] = df_input[common_cols]
                    
                    # 3. Predict
                    pIC50 = model.predict(df_ready)[0]
                    ic50_nm = 10**(9 - pIC50)
                    
                    # 4. Display Results
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted pIC50", f"{pIC50:.2f}")
                        if pIC50 > 6.0:
                            st.success("Verdict: **ACTIVE**")
                        else:
                            st.error("Verdict: **INACTIVE**")
                            
                    with col2:
                        st.metric("Estimated IC50 (nM)", f"{ic50_nm:.1f} nM")
                        # Show structure
                        mol = Chem.MolFromSmiles(user_input)
                        st.image(Chem.Draw.MolToImage(mol), caption="Query Structure", width=200)
                        
                else:
                    st.error("Invalid SMILES! Could not parse structure.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Sharav Desai & Shubhada Malode, Pharmaceutical Biotechnology Research Lab.,SCPER, KOPARGAON-423601 For academic use only.")
