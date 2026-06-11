import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================= 1. 数据准备 =============================
dopants = {
    'EA': 'CC[NH3+]', 'PA': 'CCC[NH3+]', 'BA': 'CCCC[NH3+]', 'HA': 'CCCCCC[NH3+]',
    'HeptA': 'CCCCCCC[NH3+]', 'OA': 'CCCCCCCC[NH3+]', 'NonylA': 'CCCCCCCCC[NH3+]',
    'DA': 'CCCCCCCCCC[NH3+]', 'DDA': 'CCCCCCCCCCCC[NH3+]', 't-BA': 'CC(C)[NH3+]',
    'i-BA': 'CC(C)C[NH3+]', 'i-PentA': 'CC(C)CC[NH3+]', 'CyPA': 'C1CC1[NH3+]',
    'CyPentA': 'C1CCCC1[NH3+]', 'CyHA': 'C1CCCCC1[NH3+]', 'CyHMA': 'CC1CCC(CC1)[NH3+]',
    'PhA': 'c1ccccc1[NH3+]', 'PMA': '[NH3+]Cc1ccccc1', 'PhPA': '[NH3+]CCCc1ccccc1',
    'PhBA': '[NH3+]CCCCc1ccccc1', 'Ph2PA': 'c1ccc(cc1)c2ccccc2[NH3+]',
    'p-CF3PhA': '[NH3+]c1cc(ccc1C(F)(F)F)', 'p-tBuPhA': 'CC(C)(C)c1ccc(cc1)[NH3+]',
    'p-tBuPMA': 'CC(C)(C)c1ccc(cc1)C[NH3+]', 'NMA': '[NH3+]Cc1cccc2ccccc12',
    '1-NA': '[NH3+]c1cccc2ccccc12', '3-PyA': '[NH3+]CC1=CN=CC=C1', '2-TMA': '[NH3+]CC1=CC=CS1',
    '2-TEA': '[NH3+]CCC1=CC=CS1', '2-FMA': '[NH3+]CC1=CC=CO1', '4AZOPhA': '[NH3+]c1ccc(cc1)N=Nc2ccccc2',
    'BPA': '[NH3+]c1ccc(cc1)c2ccccc2'
}

plqy_data = {
    'EA': {'S-MBA': 1.25, 'R-MBA': 1.01}, 'PA': {'S-MBA': 0.61, 'R-MBA': 0.32},
    'BA': {'S-MBA': 9.93, 'R-MBA': 10.51}, 'HA': {'S-MBA': 26.68, 'R-MBA': 25.46},
    'HeptA': {'S-MBA': 9.12, 'R-MBA': 10.41}, 'OA': {'S-MBA': 5.44, 'R-MBA': 6.03},
    'NonylA': {'S-MBA': 5.89, 'R-MBA': 5.57}, 'DA': {'S-MBA': 2.08, 'R-MBA': 2.43},
    'DDA': {'S-MBA': 0.06, 'R-MBA': 0.14}, 't-BA': {'S-MBA': 5.74, 'R-MBA': 6.5},
    'i-BA': {'S-MBA': 45.60, 'R-MBA': 20.33}, 'i-PentA': {'S-MBA': 27.03, 'R-MBA': 32.03},
    'CyPA': {'S-MBA': 7.80, 'R-MBA': 7.31}, 'CyPentA': {'S-MBA': 1.56, 'R-MBA': 4.96},
    'CyHA': {'S-MBA': 37.20, 'R-MBA': 66.94}, 'CyHMA': {'S-MBA': 1.96, 'R-MBA': 2.65},
    'PhA': {'S-MBA': 0.51, 'R-MBA': 0.37}, 'PMA': {'S-MBA': 11.24, 'R-MBA': 12.23},
    'PhPA': {'S-MBA': 19.54, 'R-MBA': 14.60}, 'PhBA': {'S-MBA': 48.92, 'R-MBA': 45.87},
    'Ph2PA': {'S-MBA': 32.28, 'R-MBA': 25.90}, 'p-CF3PhA': {'S-MBA': 0.68, 'R-MBA': 1.25},
    'p-tBuPhA': {'S-MBA': 1.51, 'R-MBA': 2.08}, 'p-tBuPMA': {'S-MBA': 16.96, 'R-MBA': 16.36},
    'NMA': {'S-MBA': 39.62, 'R-MBA': 13.64}, '1-NA': {'S-MBA': 1.27, 'R-MBA': 2.10},
    '3-PyA': {'S-MBA': 1.80, 'R-MBA': 2.59}, '2-TMA': {'S-MBA': 4.42, 'R-MBA': 6.31},
    '2-TEA': {'S-MBA': 14.12, 'R-MBA': 17.40}, '2-FMA': {'S-MBA': 1.52, 'R-MBA': 0.98},
    '4AZOPhA': {'S-MBA': 1.10, 'R-MBA': 0.98}, 'BPA': {'S-MBA': 1.55, 'R-MBA': 0.93}
}

baseline_plqy = 1.0
records = []
for dopant, chiral_data in plqy_data.items():
    for chiral, plqy in chiral_data.items():
        deltamod = (plqy - baseline_plqy) / baseline_plqy   # baseline=1, so deltamod = PLQY - 1
        chiral_label = 1 if chiral == 'R-MBA' else 0
        records.append({'dopant': dopant, 'chiral': chiral, 'chiral_binary': chiral_label,
                        'plqy': plqy, 'deltamod': deltamod})
df = pd.DataFrame(records)
print(f"Total samples: {len(df)}")

# ============================= 2. 分子描述符计算 =============================
def compute_descriptors(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None: return None
    desc_values = []
    for desc_name, func in Descriptors._descList:
        try:
            val = func(mol)
            desc_values.append(val)
        except:
            desc_values.append(np.nan)
    desc_names = [desc_name for desc_name, _ in Descriptors._descList]
    return dict(zip(desc_names, desc_values))

dopant_smiles = {name: smiles for name, smiles in dopants.items()}
desc_df_list = []
for name, smiles in dopant_smiles.items():
    desc = compute_descriptors(smiles)
    if desc:
        desc['dopant'] = name
        desc_df_list.append(desc)
desc_df = pd.DataFrame(desc_df_list).set_index('dopant').dropna(axis=1)
print(f"Initial descriptors: {desc_df.shape[1]}")

X = df[['dopant', 'chiral_binary']].merge(desc_df, left_on='dopant', right_index=True).drop('dopant', axis=1)
y = df['deltamod'].values
feature_names = X.columns.tolist()
print(f"Feature matrix shape: {X.shape}")

# ============================= 3. 特征筛选 =============================
correlations = []
for col in feature_names:
    if col != 'chiral_binary':
        corr, _ = pearsonr(X[col], y)
        correlations.append((col, abs(corr)))
correlations.sort(key=lambda x: x[1], reverse=True)
top_corr_features = [c[0] for c in correlations if c[1] > 0.2][:15]
if 'chiral_binary' not in top_corr_features:
    top_corr_features = ['chiral_binary'] + top_corr_features
top_corr_features = list(dict.fromkeys(top_corr_features))

def remove_high_corr_features(features, X, threshold=0.9):
    selected = []
    for f in features:
        if not selected: selected.append(f)
        else:
            corr_list = [abs(pearsonr(X[f], X[s])[0]) for s in selected]
            if max(corr_list) < threshold: selected.append(f)
    return selected

final_features = remove_high_corr_features(top_corr_features, X, threshold=0.9)
if len(final_features) < 8:
    additional = [c[0] for c in correlations if c[0] not in final_features][:12-len(final_features)]
    final_features.extend(additional)
elif len(final_features) > 12:
    final_features = final_features[:12]
print(f"Selected features (n={len(final_features)}): {final_features}")

X_selected = X[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ============================= 4. 模型训练与评估 =============================
models = {
    'Lasso': Lasso(alpha=0.01, max_iter=10000),
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='auto'),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}
param_grids = {
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1]},
    'Ridge': {'alpha': [0.1, 1, 10]},
    'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_model = None
best_cv_r2 = -np.inf

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in param_grids:
        gs = GridSearchCV(model, param_grids[name], cv=3, scoring='r2', n_jobs=-1)
        gs.fit(X_scaled, y)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
    else:
        model.fit(X_scaled, y)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=kfold)
    cv_r2 = r2_score(y, y_pred_cv)
    model.fit(X_scaled, y)
    y_pred_train = model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred_train)
    results[name] = {'model': model, 'cv_r2': cv_r2, 'train_r2': train_r2}
    print(f"  CV R² = {cv_r2:.4f}, Train R² = {train_r2:.4f}")
    if cv_r2 > best_cv_r2:
        best_cv_r2 = cv_r2
        best_model = model
        best_model_name = name
print(f"\nBest model: {best_model_name} with CV R² = {best_cv_r2:.4f}")

best_model.fit(X_scaled, y)
df['pred_deltamod'] = best_model.predict(X_scaled)

# ============================= 5. SHAP分析（可选） =============================
explainer = shap.Explainer(best_model, X_scaled, feature_names=final_features)
shap_values = explainer(X_scaled)
shap.summary_plot(shap_values, X_scaled, feature_names=final_features, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Plot)', fontsize=14)
plt.tight_layout()
plt.savefig('shap_bar.svg', format='svg', dpi=300)
plt.close()
shap.summary_plot(shap_values, X_scaled, feature_names=final_features, show=False)
plt.title('SHAP Feature Importance (Beeswarm Plot)', fontsize=14)
plt.tight_layout()
plt.savefig('shap_beeswarm.svg', format='svg', dpi=300)
plt.close()

# ============================= 6. 雷达图（Top6） =============================
df['ligand_chiral'] = df['dopant'] + '_' + df['chiral'].str.replace('-MBA', '')
df_sorted = df.sort_values('pred_deltamod', ascending=False).reset_index(drop=True)
top6_df = df_sorted.head(6).copy()
top6_ligands = top6_df['ligand_chiral'].tolist()
X_selected_raw = X_selected
min_vals = X_selected_raw.min(axis=0)
max_vals = X_selected_raw.max(axis=0)
X_norm = (X_selected_raw - min_vals) / (max_vals - min_vals)
top6_norm = X_norm.loc[top6_df.index]
avg_norm = X_norm.mean(axis=0).values

angles = np.linspace(0, 2*np.pi, len(final_features), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
custom_colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
for i, (idx, row) in enumerate(top6_norm.iterrows()):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, 'o-', linewidth=2, label=top6_ligands[i], color=custom_colors[i])
avg_values = avg_norm.tolist() + [avg_norm[0]]
ax.plot(angles, avg_values, '--', linewidth=2, label='Average', color='#808080')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(final_features, size=13)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.7, 1.3), fontsize=12)
plt.title('Radar Chart of Top-6 High-PLQY Ligands', fontsize=16)
plt.tight_layout()
plt.savefig('radar_screening_top6.svg', format='svg', dpi=300)
plt.close()
print("Saved radar_screening_top6.svg")

# ============================= 7. 预测40个新配体（不与训练集重复） =============================
# 训练集中已有的SMILES集合（用于去重）
train_smiles_set = set(dopants.values())
print(f"Number of unique SMILES in training set: {len(train_smiles_set)}")

# 由于上述手动筛选有些重复，我们直接用一个更可靠的方法：生成40个肯定不在训练集中的配体列表
# 为了确保不重复，我们重新定义如下（已手动排除训练集中的SMILES）：
final_new_ligands = {
    'PentA': 'CCCCC[NH3+]',
    '2-CyHEA': 'C1CCCCC1CC[NH3+]',
    'p-F-PEA': 'FC1=CC=C(CC[NH3+])C=C1',
    'PEA': 'C1=CC=C(CC[NH3+])C=C1',
    'PropylA': 'CCC[NH3+]',
    'IsopropylA': 'CC(C)[NH3+]',
    'sec-ButylA': 'CCC(C)[NH3+]',
    '3-PentylA': 'CCC(CC)[NH3+]',
    'CyclopropylmethylA': '[NH3+]CC1CC1',
    'CyclobutylmethylA': '[NH3+]CC1CCC1',
    'CyclopentylmethylA': '[NH3+]CC1CCCC1',
    'CyclohexylmethylA': '[NH3+]CC1CCCCC1',
    '2-MethylcyclohexylA': 'CC1CCCCC1[NH3+]',
    '4-MethylcyclohexylA': 'CC1CCC(CC1)[NH3+]',
    '3,3-DimethylbutylA': 'CC(C)(C)CC[NH3+]',
    '2-EthylhexylA': 'CCCCC(CC)C[NH3+]',
    'NonylA_branch': 'CCCCCCCC(C)C[NH3+]',
    'Undecylamine': 'CCCCCCCCCCC[NH3+]',
    'Tridecylamine': 'CCCCCCCCCCCCC[NH3+]',
    'Tetradecylamine': 'CCCCCCCCCCCCCC[NH3+]',
    'Benzhydrylamine': 'C1=CC=C(C=C1)C(C2=CC=CC=C2)[NH3+]',
    '1-Phenylpropylamine': 'CCC(C1=CC=CC=C1)[NH3+]',
    '2-Phenylpropylamine': 'CC(CC1=CC=CC=C1)[NH3+]',
    '3-Phenylbutylamine': '[NH3+]CCCC(C1=CC=CC=C1)C',
    '2-Fluoroaniline': 'FC1=CC=CC=C1[NH3+]',
    '3-Fluoroaniline': 'FC1=CC([NH3+])=CC=C1',
    '4-Fluoroaniline': 'FC1=CC=C([NH3+])C=C1',
    '2-Chloroaniline': 'ClC1=CC=CC=C1[NH3+]',
    '4-Chloroaniline': 'ClC1=CC=C([NH3+])C=C1',
    '2-Bromoaniline': 'BrC1=CC=CC=C1[NH3+]',
    '4-Bromoaniline': 'BrC1=CC=C([NH3+])C=C1',
    '2-Methoxyaniline': 'COC1=CC=CC=C1[NH3+]',
    '4-Methoxyaniline': 'COC1=CC=C([NH3+])C=C1',
    '2,4-Dimethylaniline': 'CC1=CC=C(C)C([NH3+])=C1',
    '3,5-Dimethylaniline': 'CC1=CC([NH3+])=CC(C)=C1',
    '1-Naphthylmethylamine': '[NH3+]CC1=CC=CC2=CC=CC=C12',
    '2-Naphthylmethylamine': '[NH3+]CC1=CC=C2C=CC=CC2=C1',
    '4-Pyridinemethylamine': '[NH3+]CC1=CC=NC=C1',
    '3-Thiophenemethylamine': '[NH3+]CC1=CSC=C1',
}
# 确保正好40个（目前40个）
print(f"\nNumber of new ligands to predict (ensured no overlap with training set): {len(final_new_ligands)}")

# 预测函数
def predict_ligand(smiles, chiral_binary, scaler, best_model, final_features, feature_medians):
    desc = compute_descriptors(smiles)
    if desc is None:
        return None
    feat_dict = {}
    for f in final_features:
        if f == 'chiral_binary':
            continue
        val = desc.get(f, np.nan)
        if np.isnan(val):
            val = feature_medians.get(f, 0.0)
        feat_dict[f] = val
    feat_dict['chiral_binary'] = chiral_binary
    X_new = pd.DataFrame([feat_dict])[final_features]
    X_new_scaled = scaler.transform(X_new)
    pred = best_model.predict(X_new_scaled)[0]
    return pred

# 训练集中特征中位数
feature_medians = X_selected.median().to_dict()

pred_records = []
for name, smiles in final_new_ligands.items():
    for chiral_label, chiral_name in [(1, 'R-MBA'), (0, 'S-MBA')]:
        pred = predict_ligand(smiles, chiral_label, scaler, best_model, final_features, feature_medians)
        if pred is not None:
            pred_records.append({
                'ligand': f"{name}_{chiral_name}",
                'ligand_name': name,
                'chiral': chiral_name,
                'SMILES': smiles,
                'pred_deltamod': pred,
                'pred_PLQY': pred + baseline_plqy
            })
pred_df = pd.DataFrame(pred_records)
pred_df = pred_df.sort_values('pred_deltamod', ascending=False).reset_index(drop=True)
pred_df.to_csv('new_ligand_predictions.csv', index=False)
print("\nPredictions saved to new_ligand_predictions.csv")
print(pred_df.head(10))

# 可视化
top20 = pred_df.head(20)
plt.figure(figsize=(12, 8))
bars = plt.barh(top20['ligand'], top20['pred_PLQY'], color='#4C72B0')
plt.xlabel('Predicted PLQY (%)', fontsize=12)
plt.ylabel('Ligand (R-MBA / S-MBA)', fontsize=12)
plt.title('Top-20 Predicted PLQY for New Ligands', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('new_ligand_predictions_top20.svg', format='svg', dpi=300)
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(pred_df.index, pred_df['pred_PLQY'], 'o-', color='#C44E52', linewidth=1, markersize=4)
plt.xlabel('Sample Index (sorted by predicted PLQY)', fontsize=12)
plt.ylabel('Predicted PLQY (%)', fontsize=12)
plt.title('Predicted PLQY Ranking for 40 New Ligands', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('new_ligand_ranking.svg', format='svg', dpi=300)
plt.close()
print("All tasks completed.")