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

# ------------------------------
# 1. 数据准备
# ------------------------------
dopants = {
    'EA': 'CC[NH3+]',
    'PA': 'CCC[NH3+]',
    'BA': 'CCCC[NH3+]',
    'HA': 'CCCCCC[NH3+]',
    'HeptA': 'CCCCCCC[NH3+]',
    'OA': 'CCCCCCCC[NH3+]',
    'NonylA': 'CCCCCCCCC[NH3+]',
    'DA': 'CCCCCCCCCC[NH3+]',
    'DDA': 'CCCCCCCCCCCC[NH3+]',
    't-BA': 'CC(C)[NH3+]',
    'i-BA': 'CC(C)C[NH3+]',
    'i-PentA': 'CC(C)CC[NH3+]',
    'CyPA': 'C1CC1[NH3+]',
    'CyPentA': 'C1CCCC1[NH3+]',
    'CyHA': 'C1CCCCC1[NH3+]',
    'CyHMA': 'CC1CCC(CC1)[NH3+]',
    'PhA': 'c1ccccc1[NH3+]',
    'PMA': '[NH3+]Cc1ccccc1',
    'PhPA': '[NH3+]CCCc1ccccc1',
    'PhBA': '[NH3+]CCCCc1ccccc1',
    'Ph2PA': 'c1ccc(cc1)c2ccccc2[NH3+]',
    'p-CF3PhA': '[NH3+]c1cc(ccc1C(F)(F)F)',
    'p-tBuPhA': 'CC(C)(C)c1ccc(cc1)[NH3+]',
    'p-tBuPMA': 'CC(C)(C)c1ccc(cc1)C[NH3+]',
    'NMA': '[NH3+]Cc1cccc2ccccc12',
    '1-NA': '[NH3+]c1cccc2ccccc12',
    '3-PyA': '[NH3+]CC1=CN=CC=C1',
    '2-TMA': '[NH3+]CC1=CC=CS1',
    '2-TEA': '[NH3+]CCC1=CC=CS1',
    '2-FMA': '[NH3+]CC1=CC=CO1',
    '4AZOPhA': '[NH3+]c1ccc(cc1)N=Nc2ccccc2',
    'BPA': '[NH3+]c1ccc(cc1)c2ccccc2'
}

plqy_data = {
    'EA': {'S-MBA': 1.25, 'R-MBA': 1.01},
    'PA': {'S-MBA': 0.61, 'R-MBA': 0.32},
    'BA': {'S-MBA': 9.93, 'R-MBA': 10.51},
    'HA': {'S-MBA': 26.68, 'R-MBA': 25.46},
    'HeptA': {'S-MBA': 9.12, 'R-MBA': 10.41},
    'OA': {'S-MBA': 5.44, 'R-MBA': 6.03},
    'NonylA': {'S-MBA': 5.89, 'R-MBA': 5.57},
    'DA': {'S-MBA': 2.08, 'R-MBA': 2.43},
    'DDA': {'S-MBA': 0.06, 'R-MBA': 0.14},
    't-BA': {'S-MBA': 5.74, 'R-MBA': 6.5},
    'i-BA': {'S-MBA': 45.60, 'R-MBA': 20.33},
    'i-PentA': {'S-MBA': 27.03, 'R-MBA': 32.03},
    'CyPA': {'S-MBA': 7.80, 'R-MBA': 7.31},
    'CyPentA': {'S-MBA': 1.56, 'R-MBA': 4.96},
    'CyHA': {'S-MBA': 37.20, 'R-MBA': 66.94},
    'CyHMA': {'S-MBA': 1.96, 'R-MBA': 2.65},
    'PhA': {'S-MBA': 0.51, 'R-MBA': 0.37},
    'PMA': {'S-MBA': 11.24, 'R-MBA': 12.23},
    'PhPA': {'S-MBA': 19.54, 'R-MBA': 14.60},
    'PhBA': {'S-MBA': 48.92, 'R-MBA': 45.87},
    'Ph2PA': {'S-MBA': 32.28, 'R-MBA': 25.90},
    'p-CF3PhA': {'S-MBA': 0.68, 'R-MBA': 1.25},
    'p-tBuPhA': {'S-MBA': 1.51, 'R-MBA': 2.08},
    'p-tBuPMA': {'S-MBA': 16.96, 'R-MBA': 16.36},
    'NMA': {'S-MBA': 39.62, 'R-MBA': 13.64},
    '1-NA': {'S-MBA': 1.27, 'R-MBA': 2.10},
    '3-PyA': {'S-MBA': 1.80, 'R-MBA': 2.59},
    '2-TMA': {'S-MBA': 4.42, 'R-MBA': 6.31},
    '2-TEA': {'S-MBA': 14.12, 'R-MBA': 17.40},
    '2-FMA': {'S-MBA': 1.52, 'R-MBA': 0.98},
    '4AZOPhA': {'S-MBA': 1.10, 'R-MBA': 0.98},
    'BPA': {'S-MBA': 1.55, 'R-MBA': 0.93}
}

baseline_plqy = 0.01
records = []
for dopant, chiral_data in plqy_data.items():
    for chiral, plqy in chiral_data.items():
        deltamod = (plqy - baseline_plqy) / baseline_plqy
        chiral_label = 1 if chiral == 'R-MBA' else 0
        records.append({
            'dopant': dopant,
            'chiral': chiral,
            'chiral_binary': chiral_label,
            'plqy': plqy,
            'deltamod': deltamod
        })
df = pd.DataFrame(records)
print(f"Total samples: {len(df)}")


# ------------------------------
# 2. 计算分子描述符 (RDKit)
# ------------------------------
def compute_descriptors(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
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
desc_df = pd.DataFrame(desc_df_list).set_index('dopant')
desc_df = desc_df.dropna(axis=1)
print(f"Initial number of descriptors: {desc_df.shape[1]}")

X = df[['dopant', 'chiral_binary']].merge(desc_df, left_on='dopant', right_index=True).drop('dopant', axis=1)
y = df['deltamod'].values
feature_names = X.columns.tolist()
print(f"Feature matrix shape: {X.shape}")

# ------------------------------
# 3. 特征筛选 (相关性 + 剔除高相关，最终保留8-12个)
# ------------------------------
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
        if not selected:
            selected.append(f)
        else:
            corr_list = [abs(pearsonr(X[f], X[s])[0]) for s in selected]
            if max(corr_list) < threshold:
                selected.append(f)
    return selected


final_features = remove_high_corr_features(top_corr_features, X, threshold=0.9)
if len(final_features) < 8:
    additional = [c[0] for c in correlations if c[0] not in final_features][:12 - len(final_features)]
    final_features.extend(additional)
elif len(final_features) > 12:
    final_features = final_features[:12]
print(f"Selected features (n={len(final_features)}): {final_features}")

X_selected = X[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ------------------------------
# 4. 特征筛选可视化（相关性图）
# ------------------------------
target_corr = []
for col in feature_names:
    if col != 'chiral_binary':
        corr, _ = pearsonr(X[col], y)
        target_corr.append((col, corr))
corr_chiral, _ = pearsonr(X['chiral_binary'], y)
target_corr.append(('chiral_binary', corr_chiral))
target_corr_df = pd.DataFrame(target_corr, columns=['Feature', 'Correlation'])
target_corr_df = target_corr_df[abs(target_corr_df['Correlation']) > 0.2].sort_values('Correlation', key=abs,
                                                                                      ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(target_corr_df['Feature'], target_corr_df['Correlation'], color='#4C72B0')
for i, (_, row) in enumerate(target_corr_df.iterrows()):
    bars[i].set_color('#C44E52' if row['Correlation'] < 0 else '#4C72B0')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Pearson Correlation with Δmod')
plt.title('Feature-Target Correlations (|r| > 0.2)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_target_correlation.svg', format='svg', dpi=300)
plt.close()
print("Saved feature_target_correlation.svg")

selected_corr = X_selected.corr(method='pearson')
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(selected_corr, dtype=bool), k=1)
sns.heatmap(selected_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Pearson Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('selected_features_correlation.svg', format='svg', dpi=300)
plt.close()
print("Saved selected_features_correlation.svg")

# ------------------------------
# 5. 模型训练与评估
# ------------------------------
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
        best_model_params = gs.best_params_
        model = gs.best_estimator_
        print(f"  Best params: {best_model_params}")
    else:
        model.fit(X_scaled, y)

    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=kfold)
    cv_r2 = r2_score(y, y_pred_cv)
    model.fit(X_scaled, y)
    y_pred_train = model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred_train)

    results[name] = {
        'model': model,
        'cv_r2': cv_r2,
        'train_r2': train_r2,
        'y_pred_cv': y_pred_cv,
        'y_pred_train': y_pred_train
    }
    print(f"  CV R² = {cv_r2:.4f}, Train R² = {train_r2:.4f}")

    if cv_r2 > best_cv_r2:
        best_cv_r2 = cv_r2
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with CV R² = {best_cv_r2:.4f}")

# ------------------------------
# 6. 各模型拟合散点图
# ------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
for i, (name, res) in enumerate(results.items()):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(y, res['y_pred_cv'], alpha=0.7, color=colors[i % len(colors)], edgecolors='w', s=80)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax1.set_xlabel('True Δmod')
    ax1.set_ylabel('Predicted Δmod (CV)')
    ax1.set_title(f'{name} - CV (R²={res["cv_r2"]:.3f})')
    ax2.scatter(y, res['y_pred_train'], alpha=0.7, color=colors[i % len(colors)], edgecolors='w', s=80)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax2.set_xlabel('True Δmod')
    ax2.set_ylabel('Predicted Δmod (Full)')
    ax2.set_title(f'{name} - Full Train (R²={res["train_r2"]:.3f})')
    plt.tight_layout()
    plt.savefig(f'{name}_scatter.svg', format='svg', dpi=300)
    plt.close()
    print(f"Saved {name}_scatter.svg")

# ------------------------------
# 7. 最佳模型 SHAP 分析
# ------------------------------
best_model.fit(X_scaled, y)
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
print("Saved SHAP plots")

# ------------------------------
# 8. 雷达图：按预测 Δmod 排名前6的配体（含手性）
# ------------------------------
# 先得到预测值并排序
df['pred_deltamod'] = best_model.predict(X_scaled)
df['ligand_chiral'] = df['dopant'] + '_' + df['chiral'].str.replace('-MBA', '')
df_sorted = df.sort_values('pred_deltamod', ascending=False).reset_index(drop=True)
# 取前6个独特的配体（可能同一配体两种手性都入选，保留两者）
top6_df = df_sorted.head(6).copy()
top6_ligands = top6_df['ligand_chiral'].tolist()
print(f"Top 6 ligands by predicted Δmod: {top6_ligands}")

# 计算每个配体的平均特征值（跨手性？不，这里我们展示每个具体样本的特征，因为手性特征已经包含）
# 直接使用 X_selected 中对应索引的特征值（已经标准化后的 X_scaled 或原始特征？为了雷达图可比性，使用原始特征值并做 Min-Max 归一化）
# 但更合理：使用原始特征值（未标准化），然后对每个特征在所有配体上做 Min-Max 归一化
X_selected_raw = X_selected  # 未标准化的原始特征
# 计算每个样本的归一化特征值（基于所有样本的 min/max）
min_vals = X_selected_raw.min(axis=0)
max_vals = X_selected_raw.max(axis=0)
X_norm = (X_selected_raw - min_vals) / (max_vals - min_vals)
# 取出前6个样本的归一化特征
top6_norm = X_norm.loc[top6_df.index]  # 按索引取
# 同时计算所有样本的平均归一化特征（作为参考）
avg_norm = X_norm.mean(axis=0).values

# 绘图
angles = np.linspace(0, 2 * np.pi, len(final_features), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
# 为每个配体分配不同颜色（使用颜色映射）
cmap = plt.cm.tab10
for i, (idx, row) in enumerate(top6_norm.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=top6_ligands[i], color=cmap(i))
# 添加平均线
avg_values = avg_norm.tolist()
avg_values += [avg_norm[0]]
ax.plot(angles, avg_values, 'k--', linewidth=2, label='Average (all samples)')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(final_features, size=7, rotation=45)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
plt.title('Radar Chart of Top-6 High-PLQY Ligands (by predicted Δmod)', fontsize=12)
plt.tight_layout()
plt.savefig('radar_screening_top6.svg', format='svg', dpi=300)
plt.close()
print("Saved radar_screening_top6.svg (top 6 ligands)")

# ------------------------------
# 9. 针对这前6个配体生成 SHAP 瀑布图（每个配体的第一个样本，即 R 构型或 S 构型，按实际数据）
# ------------------------------
for i, (idx, row) in enumerate(top6_df.iterrows()):
    dopant_name = row['ligand_chiral']
    shap.waterfall_plot(shap_values[idx], max_display=10, show=False)
    plt.title(f'SHAP Waterfall Plot for {dopant_name}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'shap_{dopant_name}_waterfall.svg', format='svg', dpi=300)
    plt.close()
print("Saved SHAP waterfall plots for top 6 ligands")

# ------------------------------
# 10. 打分标准与排序对比（横坐标为配体名称）
# ------------------------------
plt.figure(figsize=(14, 6))
plt.plot(top6_df['ligand_chiral'], top6_df['deltamod'], 'o-', color='#4C72B0', label='Experimental', linewidth=2,
         markersize=6)
plt.plot(top6_df['ligand_chiral'], top6_df['pred_deltamod'], 's-', color='#C44E52', label='Predicted', linewidth=2,
         markersize=6)
plt.xlabel('Ligand (sorted by predicted Δmod)', fontsize=12)
plt.ylabel('Δmod', fontsize=12)
plt.title('Top-6 Ligands: Predicted vs Experimental Δmod')
plt.xticks(rotation=45, fontsize=9)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ranking_comparison_top6.svg', format='svg', dpi=300)
plt.close()
print("Saved ranking comparison for top 6 ligands")

# 全部排序对比图（保留原版本，但横坐标太长，可以只保存前30个？或者使用条形图。这里简单保存前30个）
df_sorted_all = df.sort_values('pred_deltamod', ascending=False).reset_index(drop=True)
ligand_labels_all = df_sorted_all['ligand_chiral'].tolist()
plt.figure(figsize=(16, 6))
plt.plot(ligand_labels_all, df_sorted_all['deltamod'], 'o-', color='#4C72B0', label='Experimental', linewidth=1,
         markersize=4)
plt.plot(ligand_labels_all, df_sorted_all['pred_deltamod'], 's-', color='#C44E52', label='Predicted', linewidth=1,
         markersize=4)
plt.xlabel('Ligand (sorted by predicted Δmod)', fontsize=12)
plt.ylabel('Δmod', fontsize=12)
plt.title('All Ligands: Predicted vs Experimental Δmod')
plt.xticks(rotation=90, fontsize=6)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ranking_comparison_all.svg', format='svg', dpi=300)
plt.close()
print("Saved ranking comparison for all ligands")

ranking_table = df_sorted_all[['ligand_chiral', 'dopant', 'chiral', 'deltamod', 'pred_deltamod']]
ranking_table.to_csv('dopant_ranking.csv', index=False)
print("Ranking saved to dopant_ranking.csv")


# ------------------------------
# 11. 预测新配体的函数 (预留接口)
# ------------------------------
def predict_deltamod(smiles, chiral='R'):
    desc = compute_descriptors(smiles)
    if desc is None:
        raise ValueError("Invalid SMILES string")
    feat_dict = {}
    for f in final_features:
        if f == 'chiral_binary':
            continue
        val = desc.get(f, np.nan)
        if np.isnan(val):
            val = X_selected[f].median()
        feat_dict[f] = val
    chiral_binary = 1 if chiral == 'R' else 0
    feat_dict['chiral_binary'] = chiral_binary
    X_new = pd.DataFrame([feat_dict])[final_features]
    X_new_scaled = scaler.transform(X_new)
    pred = best_model.predict(X_new_scaled)[0]
    return pred


example_smiles = "CCCCCCCCCCCC[NH3+]"
example_chiral = "R"
try:
    pred_val = predict_deltamod(example_smiles, example_chiral)
    print(f"\nPrediction for new ligand ({example_smiles}, chiral={example_chiral}): Δmod = {pred_val:.3f}")
except Exception as e:
    print(f"Prediction failed: {e}")

print("\nAll tasks completed. SVG plots and CSV ranking saved.")