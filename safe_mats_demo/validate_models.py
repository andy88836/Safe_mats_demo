"""Model loading validation: tuned vs base comparison on holdout set."""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

BENZENE_SEED = 8
TOLUENE_SEED = 42

# --- Benzene ---
print("=" * 60)
print("BENZENE MODEL VALIDATION (random_state=8)")
print("=" * 60)

df_b = pd.read_csv("D:/Doc_Projects/All_des_cal/sine_matrix/Benzene/sine_matrix_benzene_IQR_2_modeling_cleaned.csv")
feature_cols_b = [c for c in df_b.columns if c.startswith("sine coulomb matrix eig")]
X_b = df_b[feature_cols_b].values
y_b = df_b["benzene_adsorption"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, random_state=BENZENE_SEED)

base_model_b = joblib.load("D:/Doc_Projects/All_des_cal/sine_matrix/Benzene/best_model_benzene.pkl")
scaler_b = joblib.load("D:/Doc_Projects/All_des_cal/sine_matrix/Benzene/scaler_benzene.pkl")
X_test_b_scaled = scaler_b.transform(X_test_b)
y_pred_base = base_model_b.predict(X_test_b_scaled)

r2_base = r2_score(y_test_b, y_pred_base)
mae_base = mean_absolute_error(y_test_b, y_pred_base)
pearson_base = pearsonr(y_test_b, y_pred_base)[0]
print(f"\nBase RF: R2={r2_base:.4f}, MAE={mae_base:.2f}, r={pearson_base:.4f}")

tuned_model_b = joblib.load("D:/Doc_Projects/All_des_cal/sine_matrix/Benzene/tuning_seed8/rf_only_seed8_best_model_benzene.pkl")
y_pred_tuned = tuned_model_b.predict(X_test_b)
r2_tuned = r2_score(y_test_b, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test_b, y_pred_tuned)
pearson_tuned = pearsonr(y_test_b, y_pred_tuned)[0]
print(f"Tuned Pipeline: R2={r2_tuned:.4f}, MAE={mae_tuned:.2f}, r={pearson_tuned:.4f}")
print(f"Delta: R2={r2_tuned-r2_base:+.4f}, MAE={mae_tuned-mae_base:+.2f}")

# --- Toluene ---
print("\n" + "=" * 60)
print("TOLUENE MODEL VALIDATION (random_state=42)")
print("=" * 60)

raw_df = pd.read_excel("D:/Doc_Projects/All_des_cal/sine_matrix/sine_matrix_toluene.xlsx")
mof_col = raw_df.columns[0]; target_col = raw_df.columns[-1]
feature_cols = raw_df.columns[1:-1]
numeric_features = raw_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
all_nan_mask = numeric_features.isna().all(axis=0)
kept_cols = feature_cols[~all_nan_mask].tolist()
cleaned_df = pd.concat([raw_df[[mof_col]], raw_df[kept_cols], raw_df[[target_col]]], axis=1)

X_t = cleaned_df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce")
y_t = pd.to_numeric(cleaned_df.iloc[:, -1], errors="coerce")
X_t = X_t.fillna(X_t.median()).values
y_t = y_t.fillna(y_t.median()).values

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=TOLUENE_SEED)

scaler_t = joblib.load("D:/Doc_Projects/All_des_cal/sine_matrix/Toluene/Toluene/scaler_toluene.pkl")
model_t = joblib.load("D:/Doc_Projects/All_des_cal/sine_matrix/Toluene/Toluene/best_model_toluene.pkl")
X_test_t_scaled = scaler_t.transform(X_test_t)
y_pred_t = model_t.predict(X_test_t_scaled)

r2_t = r2_score(y_test_t, y_pred_t)
mae_t = mean_absolute_error(y_test_t, y_pred_t)
pearson_t = pearsonr(y_test_t, y_pred_t)[0]
print(f"\nXGBoost: R2={r2_t:.4f}, MAE={mae_t:.2f}, r={pearson_t:.4f}")

# --- Summary ---
print("\n" + "=" * 60)
print("REFERENCE COMPARISON")
print("=" * 60)
print(f"Benzene tuned: R2={r2_tuned:.4f} (paper base=0.777, tuned expected ~0.786)")
print(f"Toluene:       R2={r2_t:.4f}   (paper=0.9819)")
print(f"Benzene deviation from paper base: {(r2_tuned-0.777)/0.777*100:+.1f}%")
print(f"Toluene deviation from paper:      {(r2_t-0.9819)/0.9819*100:+.2f}%")
