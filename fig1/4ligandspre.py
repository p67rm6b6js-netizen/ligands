# -*- coding: utf-8 -*-
"""
Stable ML pipeline for chiral perovskite ligand Δmod prediction
- Only classic chemical descriptors (12 per molecule)
- LASSO feature selection
- Multi-model training: Ridge, SVR, RF, GB, XGBoost
- SHAP interpretation
- Ligand ranking
- New ligand prediction with experimental validation
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------
# 1. Ligand SMILES and PLQY
# ---------------------------
ligand_smiles = {
    'EA': 'CC[NH3+]', 'PA': 'CCC[NH3+]', 'BA': 'CCCC[NH3+]', 'HA': 'CCCCCC[NH3+]',
    'HeptA': 'CCCCCCC[NH3+]', 'OA': 'CCCCCCCC[NH3+]', 'NonylA': 'CCCCCCCCC[NH3+]',
    'DA': 'CCCCCCCCCC[NH3+]', 'DDA': 'CCCCCCCCCCCC[NH3+]', 't-BA': 'CC(C)[NH3+]',
    'i-BA': 'CC(C)C[NH3+]', 'i-PentA': 'CC(C)CC[NH3+]', 'CyPA': 'C1CC1[NH3+]',
    'CyPentA': 'C1CCC1[NH3+]', 'CyHA': 'C1CCCC1[NH3+]', 'CyHMA': 'CC1CCC(CC1)[NH3+]',
    'PhA': 'c1ccccc1[NH3+]', 'PMA': 'Cc1ccccc1[NH3+]', 'PhPA': 'CCCc1ccccc1[NH3+]',
    'PhBA': 'CC(=O)c1ccccc1[NH3+]', 'Ph2PA': 'c1ccc(cc1)c2ccccc2[NH3+]',
    'p-CF3PhA': 'c1cc(ccc1C(F)(F)F)[NH3+]', 'p-tBuPhA': 'CC(C)(C)c1ccc(cc1)[NH3+]',
    'p-tBuPMA': 'CC(C)(C)c1ccc(cc1)C[NH3+]', 'NMA': 'NCC1=CC=CC2=C1C=CC=C2',
    '1-NA': 'C1=CC=CC=C1N', '3-PyA': 'NCC1=CN=CC=C1', '2-TMA': 'NCC1=CC=CS1',
    '2-TEA': 'NCCC1=CC=CS1', '2-FMA': 'NCC1=CC=CO1', '4AZOPhA': 'Nc1ccc(cc1)N=Nc2ccccc2',
    'BPA': 'c1ccc(cc1[NH3+])c2ccccc2'
}

plqy_data = {
    'EA': {'S-MBA': 1.25, 'R-MBA': 1.01}, 'PA': {'S-MBA': 0.61, 'R-MBA': 0.32},
    'BA': {'S-MBA': 9.93, 'R-MBA': 10.51}, 'HA': {'S-MBA': 26.68, 'R-MBA': 25.46},
    'HeptA': {'S-MBA': 9.12, 'R-MBA': 10.41}, 'OA': {'S-MBA': 5.44, 'R-MBA': 6.03},
    'NonylA': {'S-MBA': 5.89, 'R-MBA': 5.57}, 'DA': {'S-MBA': 2.08, 'R-MBA': 2.43},
    'DDA': {'S-MBA': 0.06, 'R-MBA': 0.14}, 't-BA': {'S-MBA': 5.74, 'R-MBA': 20.33},
    'i-BA': {'S-MBA': 45.60, 'R-MBA': 6.50}, 'i-PentA': {'S-MBA': 27.03, 'R-MBA': 32.03},
    'CyPA': {'S-MBA': 7.80, 'R-MBA': 7.31}, 'CyPentA': {'S-MBA': 1.56, 'R-MBA': 4.96},
    'CyHA': {'S-MBA': 37.20, 'R-MBA': 66.94}, 'CyHMA': {'S-MBA': 1.96, 'R-MBA': 2.65},
    'PhA': {'S-MBA': 0.51, 'R-MBA': 0.37}, 'PMA': {'S-MBA': 11.24, 'R-MBA': 12.23},
    'PhPA': {'S-MBA': 19.54, 'R-MBA': 14.60}, 'PhBA': {'S-MBA': 48.92, 'R-MBA': 45.87},
    'Ph2PA': {'S-MBA': 32.28, 'R-MBA': 25.90}, 'p-CF3PhA': {'S-MBA': 0.68, 'R-MBA': 1.25},
    'p-tBuPhA': {'S-MBA': 16.96, 'R-MBA': 16.36}, 'p-tBuPMA': {'S-MBA': 1.51, 'R-MBA': 2.08},
    'NMA': {'S-MBA': 39.62, 'R-MBA': 13.64}, '1-NA': {'S-MBA': 1.27, 'R-MBA': 2.10},
    '3-PyA': {'S-MBA': 1.10, 'R-MBA': 0.98}, '2-TMA': {'S-MBA': 4.42, 'R-MBA': 6.31},
    '2-TEA': {'S-MBA': 1.52, 'R-MBA': 0.98}, '2-FMA': {'S-MBA': 14.12, 'R-MBA': 17.40},
    '4AZOPhA': {'S-MBA': 1.80, 'R-MBA': 2.59}, 'BPA': {'S-MBA': 1.55, 'R-MBA': 0.93}
}

# 新配体的实验数据
new_ligand_experimental = {
    'PentA': {'S-MBA': 19.79, 'R-MBA': 16.28},
    '2-CyHEA': {'S-MBA': 1.33, 'R-MBA': 1.57},
    'p-F-PEABr': {'S-MBA': 8.62, 'R-MBA': 10.73},
    '2-NMABr': {'S-MBA': 9.81, 'R-MBA': 12.3}
}

mba_smiles = {'R-MBA': 'C[C@H](N)c1ccccc1', 'S-MBA': 'C[C@@H](N)c1ccccc1'}
plqy_baseline = 1.0

# 描述符名称映射
descriptor_names = [
    'Molecular Weight',
    'LogP (Hydrophobicity)',
    'TPSA (Polar Surface Area)',
    'H-Bond Acceptors',
    'H-Bond Donors',
    'Rotatable Bonds',
    'Ring Count',
    'Heavy Atom Count',
    'Fraction SP3 Carbon',
    'Molar Refractivity',
    'Balaban J Index',
    'Bertz Complexity'
]

# ---------------------------
# 2. Build dataset
# ---------------------------
data_list = []
for ligand, l_smile in ligand_smiles.items():
    for mba_type, mba_smile in mba_smiles.items():
        delta_mod = plqy_data[ligand][mba_type] - plqy_baseline
        data_list.append({
            'ligand': ligand,
            'mba_type': mba_type,
            'ligand_smiles': l_smile,
            'mba_smiles': mba_smile,
            'Δmod': delta_mod
        })
df = pd.DataFrame(data_list)


# ---------------------------
# 3. Descriptor extraction (only ligand descriptors)
# ---------------------------
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol)
    ])


# 只提取配体描述符
features_list = []
for _, row in df.iterrows():
    l_desc = get_descriptors(row['ligand_smiles'])
    features_list.append(l_desc)

X = np.array(features_list)
y = df['Δmod'].values

# 使用描述性特征名称
feature_names = descriptor_names
X = pd.DataFrame(X, columns=feature_names)

# ---------------------------
# 4. Feature selection & scaling
# ---------------------------
# 提高方差阈值，减少特征数量
var_filter = VarianceThreshold(threshold=0.05)
X_filtered = var_filter.fit_transform(X)
selected_features = X.columns[var_filter.get_support()]
X = pd.DataFrame(X_filtered, columns=selected_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 5. Train models + CV + scatter plots (使用交叉验证预测)
# ---------------------------
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': LassoCV(cv=5, max_iter=100000, random_state=42),
    'SVR': SVR(C=100, gamma=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=500, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=500, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=500, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# 创建图形目录
os.makedirs("figures", exist_ok=True)
os.makedirs("figures/model_scatter", exist_ok=True)

print("=" * 60)
print("Starting model training and evaluation...")
print("=" * 60)

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")

    # 使用交叉验证预测（得到每个样本的测试集预测值）
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=kf, method='predict')

    # 计算交叉验证的 R²
    r2_cv = r2_score(y, y_pred_cv)

    # 计算 RMSE 和 MAE
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae_cv = mean_absolute_error(y, y_pred_cv)

    # 使用 cross_val_score 获取平均 R² 和标准差
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    results[name] = scores

    print(f"{name}:")
    print(f"  CV R2 = {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Cross-validated R2 (from predictions) = {r2_cv:.3f}")
    print(f"  RMSE = {rmse_cv:.3f}")
    print(f"  MAE = {mae_cv:.3f}")

    # 绘制散点图（使用交叉验证预测值）
    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred_cv, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 添加对角线
    min_val = min(min(y), min(y_pred_cv))
    max_val = max(max(y), max(y_pred_cv))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('Experimental Δmod', fontsize=12)
    plt.ylabel('Predicted Δmod (Cross-Validation)', fontsize=12)
    plt.title(f'{name} Regression\nCV R² = {r2_cv:.3f} (Mean ± Std: {scores.mean():.3f} ± {scores.std():.3f})',
              fontsize=14)
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"figures/model_scatter/{name}_scatter_CV.svg", format="svg", dpi=300)
    plt.close()

# 选择最佳模型
best_model_name = max(results, key=lambda k: results[k].mean())
best_model = models[best_model_name]

# 用全部数据训练最佳模型（用于后续预测）
best_model.fit(X_scaled, y)
print(f"\n{'=' * 60}")
print(f"Best model: {best_model_name}")
print(f"Best model CV R2 = {results[best_model_name].mean():.3f} ± {results[best_model_name].std():.3f}")
print("=" * 60)

# ---------------------------
# 6. SHAP analysis (使用描述性名称)
# ---------------------------
print("\nGenerating SHAP plots...")
os.makedirs("figures/SHAP", exist_ok=True)

if best_model_name == 'SVR':
    background_idx = np.random.choice(X_scaled.shape[0], min(20, X_scaled.shape[0]), replace=False)
    background = X_scaled[background_idx]
    explainer = shap.KernelExplainer(best_model.predict, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)
else:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled)

# SHAP 条形图（特征重要性）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=12)
plt.title(f"SHAP Feature Importance ({best_model_name})", fontsize=14)
plt.tight_layout()
plt.savefig("figures/SHAP/SHAP_bar.svg", format="svg", dpi=300)
plt.close()
print("  ✓ Saved: figures/SHAP/SHAP_bar.svg")

# SHAP 蜂群图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False, max_display=12)
plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=14)
plt.tight_layout()
plt.savefig("figures/SHAP/SHAP_beeswarm.svg", format="svg", dpi=300)
plt.close()
print("  ✓ Saved: figures/SHAP/SHAP_beeswarm.svg")

# ---------------------------
# 7. Ligand ranking
# ---------------------------
print("\nGenerating ligand ranking...")
df['Predicted_Δmod'] = best_model.predict(X_scaled)
ligand_score = df.groupby('ligand')['Predicted_Δmod'].mean().sort_values(ascending=False)
ligand_score.to_csv("figures/Ligand_Score.csv")
print(f"  ✓ Saved: figures/Ligand_Score.csv")

plt.figure(figsize=(14, 7))
bars = plt.bar(ligand_score.index, ligand_score.values, color='steelblue', edgecolor='black')
plt.xticks(rotation=90, fontsize=10)
plt.ylabel("Predicted Δmod", fontsize=12)
plt.title("Ligand Performance Ranking", fontsize=14)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("figures/Ligand_Score.svg", format="svg", dpi=300)
plt.close()
print("  ✓ Saved: figures/Ligand_Score.svg")

# ---------------------------
# 8. R-MBA vs S-MBA comparison
# ---------------------------
print("\nGenerating R-MBA vs S-MBA comparison...")
pivot = df.pivot(index='ligand', columns='mba_type', values='Δmod')
plt.figure(figsize=(12, 6))
pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.ylabel("Δmod", fontsize=12)
plt.title("R-MBA vs S-MBA Δmod Comparison", fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.legend(title="MBA Type", fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/R_S_MBA_Comparison.svg", format="svg", dpi=300)
plt.close()
print("  ✓ Saved: figures/R_S_MBA_Comparison.svg")


# ---------------------------
# 9. New ligand prediction function with experimental comparison
# ---------------------------
def predict_new_ligand_with_exp(smiles, ligand_name, mba_type='R-MBA'):
    """预测新配体的Δmod值，并与实验值对比"""
    try:
        l_desc = get_descriptors(smiles)
    except ValueError as e:
        print(f"Error: {e}")
        return None

    # 转换为DataFrame
    x_new = pd.DataFrame([l_desc], columns=descriptor_names)

    # 应用相同的特征选择和缩放
    x_filtered = var_filter.transform(x_new)
    x_scaled = scaler.transform(x_filtered)

    # 预测
    pred = best_model.predict(x_scaled)[0]

    # 获取实验值
    exp_value = new_ligand_experimental.get(ligand_name, {}).get(mba_type, None)

    print(f"\n{'=' * 50}")
    print(f"New Ligand Prediction: {ligand_name} ({mba_type})")
    print(f"{'=' * 50}")
    print(f"SMILES: {smiles}")
    print(f"Predicted Δmod: {pred:.2f}")
    if exp_value is not None:
        error = pred - exp_value
        abs_error = abs(error)
        percent_error = (abs_error / exp_value * 100) if exp_value != 0 else float('inf')
        print(f"Experimental Δmod: {exp_value:.2f}")
        print(f"Absolute Error: {abs_error:.2f}")
        print(f"Percent Error: {percent_error:.1f}%")

        # 判断预测是否可靠
        if percent_error < 20:
            reliability = "High"
        elif percent_error < 50:
            reliability = "Medium"
        else:
            reliability = "Low"
        print(f"Prediction Reliability: {reliability}")
    print(f"{'=' * 50}\n")

    # SHAP解释
    if best_model_name == 'SVR':
        explainer_new = shap.KernelExplainer(best_model.predict, X_scaled[
            np.random.choice(X_scaled.shape[0], min(20, X_scaled.shape[0]), replace=False)])
        shap_val = explainer_new.shap_values(x_scaled, nsamples=100)
    else:
        explainer_new = shap.TreeExplainer(best_model)
        shap_val = explainer_new.shap_values(x_scaled)

    # 可视化SHAP值
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_val, pd.DataFrame(x_filtered, columns=selected_features),
                      plot_type="bar", show=False, max_display=12)
    plt.title(f"SHAP Explanation for {ligand_name} ({mba_type})\nPredicted: {pred:.2f}, Experimental: {exp_value:.2f}",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/SHAP/SHAP_{ligand_name}_{mba_type}.svg", format="svg", dpi=300)
    plt.close()

    return pred, exp_value


# ---------------------------
# 10. 预测指定的新配体并与实验值对比
# ---------------------------
print("\n" + "=" * 60)
print("Predicting new ligands with experimental validation...")
print("=" * 60)

# 定义新配体的SMILES
new_ligands = {
    'PentA': 'CCCCC[NH3+]',
    '2-CyHEA': 'C1CCCCC1CC[NH3+]',
    'p-F-PEABr': 'Fc1ccc(cc1)CC[NH3+]',
    '2-NMABr': 'c1ccc2ccccc2c1C[NH3+]'
}

# 创建结果DataFrame
new_results = []

# 预测每种MBA类型
for mba_type in ['R-MBA', 'S-MBA']:
    print(f"\n{'=' * 50}")
    print(f"Predictions for {mba_type}")
    print(f"{'=' * 50}")

    for ligand_name, smiles in new_ligands.items():
        pred_value, exp_value = predict_new_ligand_with_exp(smiles, ligand_name, mba_type=mba_type)
        new_results.append({
            'Ligand': ligand_name,
            'MBA_Type': mba_type,
            'SMILES': smiles,
            'Predicted_Δmod': pred_value,
            'Experimental_Δmod': exp_value,
            'Absolute_Error': abs(pred_value - exp_value) if exp_value is not None else None,
            'Percent_Error': (abs(pred_value - exp_value) / exp_value * 100) if exp_value and exp_value != 0 else None
        })

# 保存预测结果
new_results_df = pd.DataFrame(new_results)
new_results_df.to_csv("figures/New_Ligand_Predictions_with_Exp.csv", index=False)
print(f"\n✓ Saved: figures/New_Ligand_Predictions_with_Exp.csv")

# 计算总体误差
overall_mae = new_results_df['Absolute_Error'].mean()
overall_percent_error = new_results_df['Percent_Error'].mean()
print(f"\nOverall Performance on New Ligands:")
print(f"  Mean Absolute Error: {overall_mae:.2f}")
print(f"  Mean Percent Error: {overall_percent_error:.1f}%")

# 创建对比图：实验值 vs 预测值
plt.figure(figsize=(12, 10))

# 子图1：柱状图对比
plt.subplot(2, 2, 1)
x_pos = np.arange(len(new_ligands))
bar_width = 0.35

for i, mba_type in enumerate(['R-MBA', 'S-MBA']):
    ligand_data = new_results_df[new_results_df['MBA_Type'] == mba_type]
    pred_values = [ligand_data[ligand_data['Ligand'] == ligand]['Predicted_Δmod'].values[0] for ligand in
                   new_ligands.keys()]
    exp_values = [ligand_data[ligand_data['Ligand'] == ligand]['Experimental_Δmod'].values[0] for ligand in
                  new_ligands.keys()]

    plt.bar(x_pos + i * bar_width, pred_values, bar_width, label=f'{mba_type} Predicted', alpha=0.7)
    plt.bar(x_pos + i * bar_width, exp_values, bar_width, label=f'{mba_type} Experimental', alpha=0.3,
            edgecolor='black', linewidth=2)

plt.xlabel('Ligand', fontsize=12)
plt.ylabel('Δmod', fontsize=12)
plt.title('Predicted vs Experimental Δmod', fontsize=14)
plt.xticks(x_pos + bar_width / 2, list(new_ligands.keys()), rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

# 子图2：散点图（预测值 vs 实验值）
plt.subplot(2, 2, 2)
for mba_type in ['R-MBA', 'S-MBA']:
    ligand_data = new_results_df[new_results_df['MBA_Type'] == mba_type]
    plt.scatter(ligand_data['Experimental_Δmod'], ligand_data['Predicted_Δmod'],
                label=mba_type, alpha=0.7, s=100, edgecolors='black')

# 添加对角线
min_val = min(new_results_df[['Predicted_Δmod', 'Experimental_Δmod']].min())
max_val = max(new_results_df[['Predicted_Δmod', 'Experimental_Δmod']].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

plt.xlabel('Experimental Δmod', fontsize=12)
plt.ylabel('Predicted Δmod', fontsize=12)
plt.title('Prediction Accuracy', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 子图3：误差分析
plt.subplot(2, 2, 3)
for i, ligand_name in enumerate(new_ligands.keys()):
    ligand_data = new_results_df[new_results_df['Ligand'] == ligand_name]
    r_mba_error = ligand_data[ligand_data['MBA_Type'] == 'R-MBA']['Absolute_Error'].values[0]
    s_mba_error = ligand_data[ligand_data['MBA_Type'] == 'S-MBA']['Absolute_Error'].values[0]

    plt.bar(i * 2, r_mba_error, width=0.8, label='R-MBA Error' if i == 0 else None, color='steelblue')
    plt.bar(i * 2 + 1, s_mba_error, width=0.8, label='S-MBA Error' if i == 0 else None, color='lightcoral')

plt.xticks([i * 2 + 0.5 for i in range(len(new_ligands))], list(new_ligands.keys()), rotation=45, fontsize=10)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Absolute Error by Ligand', fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

# 子图4：百分比误差
plt.subplot(2, 2, 4)
for i, ligand_name in enumerate(new_ligands.keys()):
    ligand_data = new_results_df[new_results_df['Ligand'] == ligand_name]
    r_mba_percent = ligand_data[ligand_data['MBA_Type'] == 'R-MBA']['Percent_Error'].values[0]
    s_mba_percent = ligand_data[ligand_data['MBA_Type'] == 'S-MBA']['Percent_Error'].values[0]

    plt.bar(i * 2, r_mba_percent, width=0.8, label='R-MBA % Error' if i == 0 else None, color='steelblue')
    plt.bar(i * 2 + 1, s_mba_percent, width=0.8, label='S-MBA % Error' if i == 0 else None, color='lightcoral')

plt.xticks([i * 2 + 0.5 for i in range(len(new_ligands))], list(new_ligands.keys()), rotation=45, fontsize=10)
plt.ylabel('Percent Error (%)', fontsize=12)
plt.title('Percentage Error by Ligand', fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/New_Ligand_Comprehensive_Analysis.svg", format="svg", dpi=300)
plt.close()
print(f"✓ Saved: figures/New_Ligand_Comprehensive_Analysis.svg")

# 创建外推性分析图
plt.figure(figsize=(12, 8))

# 分析特征空间
for i, ligand_name in enumerate(new_ligands.keys()):
    ligand_smiles = new_ligands[ligand_name]
    l_desc = get_descriptors(ligand_smiles)
    x_new = pd.DataFrame([l_desc], columns=descriptor_names)
    x_filtered = var_filter.transform(x_new)

    # 计算与训练集的距离
    distances = np.linalg.norm(X_scaled - x_filtered, axis=1)
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)

    print(f"\n{ligand_name} extrapolation analysis:")
    print(f"  Minimum distance to training set: {min_distance:.3f}")
    print(f"  Mean distance to training set: {mean_distance:.3f}")

    # 判断外推性
    if min_distance < 1.0:
        extrapolation = "Low (interpolation)"
    elif min_distance < 2.0:
        extrapolation = "Medium"
    else:
        extrapolation = "High (extrapolation)"
    print(f"  Extrapolation level: {extrapolation}")

# 保存外推性分析结果
extrapolation_df = pd.DataFrame({
    'Ligand': list(new_ligands.keys()),
    'Min_Distance': [np.min(np.linalg.norm(
        X_scaled - var_filter.transform(pd.DataFrame([get_descriptors(smiles)], columns=descriptor_names)), axis=1))
                     for smiles in new_ligands.values()],
    'Mean_Distance': [np.mean(np.linalg.norm(
        X_scaled - var_filter.transform(pd.DataFrame([get_descriptors(smiles)], columns=descriptor_names)), axis=1))
                      for smiles in new_ligands.values()]
})

extrapolation_df.to_csv("figures/Extrapolation_Analysis.csv", index=False)
print(f"\n✓ Saved: figures/Extrapolation_Analysis.csv")

print("\n" + "=" * 60)
print("All figures and CSV files saved in 'figures/' folder.")
print("Model scatter plots saved in 'figures/model_scatter/' folder.")
print("SHAP plots saved in 'figures/SHAP/' folder.")
print("=" * 60)
print("\nSummary of new ligand predictions:")
print("-" * 80)
print(f"{'Ligand':<12} {'MBA Type':<10} {'Predicted':<10} {'Experimental':<12} {'Abs Error':<10} {'% Error':<10}")
print("-" * 80)
for _, row in new_results_df.iterrows():
    print(
        f"{row['Ligand']:<12} {row['MBA_Type']:<10} {row['Predicted_Δmod']:<10.2f} {row['Experimental_Δmod']:<12.2f} {row['Absolute_Error']:<10.2f} {row['Percent_Error']:<10.1f}%")
print("-" * 80)
# ============================
# 11. Predict ALL your original ligands
# ============================

print("\n" + "=" * 60)
print("Predicting all original ligands from your SMILES dictionary...")
print("=" * 60)

# ---- 你最开始发我的配体 SMILES ----
original_ligand_smiles = {
    'MA': 'C[NH3+]',
    'EA': 'CC[NH3+]',
    'PA': 'CCC[NH3+]',
    'BA': 'CCCC[NH3+]',
    'PentA': 'CCCCC[NH3+]',
    'HA': 'CCCCCC[NH3+]',
    'HeptA': 'CCCCCCC[NH3+]',
    'OA': 'CCCCCCCC[NH3+]',
    'NonylA': 'CCCCCCCCC[NH3+]',
    'DA': 'CCCCCCCCCC[NH3+]',
    'UA': 'CCCCCCCCCCC[NH3+]',
    'DDA': 'CCCCCCCCCCCC[NH3+]',
    'TDA': 'CCCCCCCCCCCCC[NH3+]',
    'IPA': 'C(C)C[NH3+]',
    'IBA': 'C(C)CC[NH3+]',
    'TBA': 'C(C)(C)C[NH3+]',
    'CPA': 'C1CC1[NH3+]',
    'CBA': 'C1CCC1[NH3+]',
    'CPentA': 'C1CCCC1[NH3+]',
    'CHA': 'C1CCCCC1[NH3+]',
    'CHMA': 'C1CCCCC1C[NH3+]',
    '2-CyHEA': 'C1CCCCC1CC[NH3+]',
    '2-Methylbutylammonium': '[NH3+]CC(C)CC',
    'Anilinium': '[NH3+]c1ccccc1',
    'BhA': '[NH3+]Cc1ccccc1',
    'PEA': '[NH3+]CCc1ccccc1',
    'PPENTA': '[NH3+]CCCCCc1ccccc1',
    '2-F-BZA': '[NH3+]Cc1ccccc1F',
    '3-F-BZA': '[NH3+]Cc1cccc(F)c1',
    '4-F-BZA': '[NH3+]Cc1ccc(F)cc1',
    '2-F-PEA': '[NH3+]CCc1ccccc1F',
    '3-F-PEA': '[NH3+]CCc1cccc(F)c1',
    '4-CF3-BZA': '[NH3+]Cc1ccc(C(F)(F)F)cc1',
    'p-F-PEABr': 'Fc1ccc(cc1)CC[NH3+]',
    'p-MeO-BZA': '[NH3+]Cc1ccc(OC)cc1',
    '3-Cl-BZA': '[NH3+]Cc1cccc(Cl)c1',
    'POEA': '[NH3+]CCOc1ccccc1',
    'BPEA': '[NH3+]CCc1ccc(-c2ccccc2)cc1',
    '2-AMPY': '[NH3+]Cc1ccccn1',
    'GA': '[NH3+]C(=[NH2+])N'
}

# ---- 预测 ----
records = []

for ligand, smile in original_ligand_smiles.items():
    try:
        desc = get_descriptors(smile)
    except Exception as e:
        print(f"[SKIP] {ligand}: {e}")
        continue

    X_new = pd.DataFrame([desc], columns=descriptor_names)
    X_new_filtered = var_filter.transform(X_new)
    X_new_scaled = scaler.transform(X_new_filtered)

    for mba in ['R-MBA', 'S-MBA']:
        pred = best_model.predict(X_new_scaled)[0]
        records.append({
            'Ligand': ligand,
            'MBA_Type': mba,
            'SMILES': smile,
            'Predicted_Δmod': pred
        })

df_pred = pd.DataFrame(records)

# ---- 保存 ----
out_dir = "figures/Predicted_Ligands"
os.makedirs(out_dir, exist_ok=True)
df_pred.to_csv(f"{out_dir}/All_Ligand_Predictions.csv", index=False)

print(f"✓ Saved: {out_dir}/All_Ligand_Predictions.csv")

# ============================
# 12. Ligand ranking (averaged over R/S)
# ============================

ranking = (
    df_pred
    .groupby('Ligand')['Predicted_Δmod']
    .mean()
    .sort_values(ascending=False)
)

ranking.to_csv(f"{out_dir}/Ligand_Ranking.csv")

plt.figure(figsize=(14, 6))
ranking.head(20).plot(kind='bar')
plt.ylabel("Average Predicted Δmod")
plt.title("Top 20 Ligands by Predicted Chiral Emission Enhancement")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{out_dir}/Top20_Ligand_Ranking.svg", format="svg", dpi=300)
plt.close()

print(f"✓ Saved ligand ranking")

# ============================
# 13. R-MBA vs S-MBA Δmod
# ============================

pivot = df_pred.pivot(index='Ligand', columns='MBA_Type', values='Predicted_Δmod')

plt.figure(figsize=(14, 6))
pivot.plot(kind='bar', figsize=(14, 6))
plt.ylabel("Predicted Δmod")
plt.title("Predicted R-MBA vs S-MBA Δmod")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{out_dir}/R_vs_S_Predicted.svg", format="svg", dpi=300)
plt.close()

print("✓ Saved R vs S comparison")

print("\n" + "=" * 60)
print("✅ All original ligands predicted successfully.")
print("=" * 60)