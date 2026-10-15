import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_auc_score

print("Generating benchmark multi-site EEG spectral power dataset...")
np.random.seed(42)

n_subjects = 300
n_channels = 64
n_bands = 5  # Delta, Theta, Alpha, Beta, Gamma
n_features = n_channels * n_bands
n_sites = 5

# Generate synthetic spectral features with site-specific baseline offsets
y = np.random.choice([0, 1], size=n_subjects, p=[0.5, 0.5])
sites = np.random.choice([f"Site_{i}" for i in range(1, n_sites + 1)], size=n_subjects)

X = np.zeros((n_subjects, n_features))
for i in range(n_subjects):
    site_offset = int(sites[i].split("_")[1]) * 0.15
    dx_signal = 0.25 if y[i] == 1 else 0.0
    X[i] = np.random.randn(n_features) + site_offset + dx_signal

print(f"Subjects: {n_subjects} | Features: {n_features} (64 channels x 5 bands) | Sites: {n_sites}")

# Evaluation 1: Standard Stratified K-Fold (Ignores Site Boundaries)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

auc_kfold = []
for train_idx, test_idx in skf.split(X, y):
    clf_rf.fit(X[train_idx], y[train_idx])
    preds = clf_rf.predict_proba(X[test_idx])[:, 1]
    auc_kfold.append(roc_auc_score(y[test_idx], preds))

mean_auc_kfold = np.mean(auc_kfold)

# Evaluation 2: Leave-One-Site-Out (LOSO) Cross-Validation
logo = LeaveOneGroupOut()
auc_loso = []
site_names = []

for train_idx, test_idx in logo.split(X, y, groups=sites):
    test_site = sites[test_idx][0]
    clf_rf.fit(X[train_idx], y[train_idx])
    preds = clf_rf.predict_proba(X[test_idx])[:, 1]
    score = roc_auc_score(y[test_idx], preds)
    auc_loso.append(score)
    site_names.append(test_site)

mean_auc_loso = np.mean(auc_loso)

print("\n--- EEG Classification Results ---")
print(f"Standard 5-Fold CV ROC-AUC (Random Split): {mean_auc_kfold:.3f}")
print(f"Leave-One-Site-Out CV ROC-AUC (Cross-Site): {mean_auc_loso:.3f}")

# Save CSV
df_res = pd.DataFrame({"Test_Site": site_names, "LOSO_ROC_AUC": auc_loso})
df_res.to_csv("reports/loso_eeg_results.csv", index=False)

# Bar chart output
plt.figure(figsize=(7, 4))
plt.bar(["Random 5-Fold CV", "Leave-One-Site-Out CV"], [mean_auc_kfold, mean_auc_loso], color=['#2b5c8f', '#e74c3c'])
plt.ylabel("Mean ROC-AUC")
plt.ylim(0.4, 1.0)
plt.title("EEG Spectral Power Classification: Random Split vs Cross-Site Generalization")
plt.tight_layout()
plt.savefig("reports/eeg_generalization.png", dpi=150)
print("Plot saved to reports/eeg_generalization.png")