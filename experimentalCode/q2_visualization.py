import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# 解决中文显示和负号问题
mpl.rcParams['axes.unicode_minus'] = False

print("正在读取数据...")
# 路径修正：直接在当前目录下找 data
df_chem = pd.read_csv('data/raw/1.csv')
df_sens = pd.read_csv('data/raw/7.csv')

# 合并数据
df = pd.merge(df_chem, df_sens, on='beer', suffixes=('_chem', '_sens'))
print(f"数据读取成功，共 {len(df)} 行")

# 确保保存图片的文件夹存在
os.makedirs('results/figures', exist_ok=True)

# ================= 画图 1：相关性热图 =================
chem_cols = ['ethyl_acetate', 'ethyl_decanoate', 'linalool', 'ethanol..v.v.', 'bitterness', 'lactic_acid.mg.L.']
sens_cols = ['aroma_fruity', 'aroma_citrus', 'taste_bitter', 'taste_sour', 'aroma_hoppy']
valid_chem = [c for c in chem_cols if c in df.columns]
valid_sens = [c for c in sens_cols if c in df.columns]

corr_matrix = df[valid_chem + valid_sens].corr().loc[valid_chem, valid_sens]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f', linewidths=.5)
plt.title('Correlation between Chemical Parameters and Sensory Descriptors', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/fig1_heatmap.png', dpi=300) # 路径修正
print("图1生成完毕：results/figures/fig1_heatmap.png")
plt.close()

# ================= 画图 2：酒精在不同啤酒风格中的箱型图 =================
plt.figure(figsize=(12, 6))
sns.boxplot(x='tasting_category_fine_chem', y='ethanol..v.v.', data=df, palette='Set3')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Ethanol across Different Beer Styles', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/fig2_boxplot_styles.png', dpi=300)
print("图2生成完毕：results/figures/fig2_boxplot_styles.png")
plt.close()

# ================= 画图 3：风味雷达图 =================
radar_cols = ['aroma_fruity', 'taste_bitter', 'taste_sweet', 'taste_sour', 'aroma_hoppy', 'aroma_malt']
valid_radar = [c for c in radar_cols if c in df.columns]
style_col = 'tasting_category_fine_chem'
styles_to_plot = ['Pils/Lager', 'Tripel', 'Kriek']
samples = df[df[style_col].isin(styles_to_plot)].groupby(style_col).first()

angles = np.linspace(0, 2*np.pi, len(valid_radar), endpoint=False).tolist()
angles += angles[:1] 

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for style in styles_to_plot:
    if style in samples.index:
        values = samples.loc[style, valid_radar].tolist()
        values += values[:1] 
        ax.plot(angles, values, linewidth=2, label=style)
        ax.fill(angles, values, alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], valid_radar, color='grey', size=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Sensory Profile (Flavor Fingerprint)', size=15, y=1.1)
plt.tight_layout()
plt.savefig('results/figures/fig3_radar_chart.png', dpi=300)
print("图3生成完毕：results/figures/fig3_radar_chart.png")
plt.close()

print("全部画图任务完成！图片已保存在 results/figures 文件夹中。")