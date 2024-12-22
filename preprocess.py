import pandas as pd
from sklearn.preprocessing import LabelEncoder
from miceforest import ImputationKernel
from imblearn.over_sampling import SMOTE


file='AD_dataset.csv'

all_features=['T1_score', 'fdg_score','sex', 'education_level', 'apoe4', 'apoe_gen1',
       'apoe_gen2', 'age', 'MMSE', 'cdr_sb', 'cdr_global', 'adas11', 'adas13',
       'adas_memory', 'adas_language', 'adas_concentration', 'adas_praxis',
       'ravlt_immediate', 'moca', 'TMT_A', 'TMT_B', 'dsst', 'logmem_delay',
       'logmem_imm', 'adni_ventricles_vol', 'adni_hippocampus_vol',
       'adni_brain_vol', 'adni_entorhinal_vol', 'adni_fusiform_vol',
       'adni_midtemp_vol', 'adni_icv', 'adni_fdg', 'adni_pib', 'adni_av45',
       'adni_abeta', 'adni_tau', 'adni_ptau']
data_models = {
          # Models using only demographic and clinical data

          "base": ["sex", "education_level", "MMSE", "cdr_sb"],

          "base_logmem": ["sex", "education_level", "MMSE", "cdr_sb", "logmem_delay", "logmem_imm"],

          "base_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_logmem_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "logmem_delay",
                                "logmem_imm"],

          "base_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis"],

          "base_ravlt_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis", "ravlt_immediate"],

          # Models including APOE

          "base_ravlt_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_adas_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                             "adas_concentration", "adas_praxis"],

          "base_ravlt_adas_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                   "adas_concentration", "adas_praxis", "ravlt_immediate"],

          # Models including imaging scores

          "base_T1score": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score"],

          "base_fdgscore": ["sex", "education_level", "MMSE", "cdr_sb", "fdg_score"],

          "base_scores": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score", "fdg_score"],

          "base_ravlt_scores": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "T1_score", "fdg_score"],

          "base_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                               "adas_concentration", "adas_praxis", "T1_score", "fdg_score"],

          "base_adas_memtest_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                       "adas_concentration", "adas_praxis", "ravlt_immediate", "T1_score", "fdg_score"],
       
          "all_features": all_features}

def impute_missing_values(data):
    """
    Impute missing values using MiceForest.
    """
    imputer = ImputationKernel(data, random_state=42, save_all_iterations_data=True)
    imputer.mice(5)
    return imputer.complete_data()

def apply_smote(data, target_col):
    """
    Apply SMOTE to balance the dataset.
    """
    sm = SMOTE(sampling_strategy="not majority")
    y = data[target_col]
    x = data.drop(columns=[target_col])
    X_resampled, y_resampled = sm.fit_resample(x, y)
    data = pd.concat([X_resampled, y_resampled], axis=1)
    return data

def preprocess(data):
    """
    Load and preprocess the dataset.
    """
    data['adni_abeta'] = data['adni_abeta'].replace('>1700', '1700').astype(float)
    data = data.drop(columns=['participant_id', 'session_id', 'marital_status'])
    data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])
    return data

def load_data(file_path='AD_raw_data.csv'):
    """
    Load the dataset.
    """
    data = pd.read_csv(file_path)
    data = preprocess(data)
    data = impute_missing_values(data)
    data = apply_smote(data, 'diagnosis')
    return data

