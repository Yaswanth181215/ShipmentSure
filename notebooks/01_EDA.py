"""
01_EDA.py
---------
Exploratory Data Analysis for ShipmentSure
Run this script to generate all EDA visualizations.
(Can also be used as a Jupyter notebook — copy cells)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/Train.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# ─────────────────────────────────────────────
# 1. TARGET DISTRIBUTION (Class Imbalance Check)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
sns.countplot(data=df, x='Reached.on.Time_Y.N', palette='Set2', ax=axes[0])
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('0 = Delayed  |  1 = On Time')
axes[0].set_ylabel('Count')
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=12)

# Pie chart
counts = df['Reached.on.Time_Y.N'].value_counts()
axes[1].pie(counts, labels=['Delayed', 'On Time'], autopct='%1.1f%%',
            colors=['#ff6b6b', '#51cf66'], startangle=90)
axes[1].set_title('Class Balance', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/eda_target_distribution.png", dpi=150)
plt.show()
print("✅ Target distribution saved.")

# ─────────────────────────────────────────────
# 2. NUMERICAL FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────
num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
            'Prior_purchases', 'Discount_offered', 'Weight_in_gms']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Reached.on.Time_Y.N', kde=True,
                 palette='Set2', ax=axes[i], alpha=0.7)
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].legend(['Delayed', 'On Time'], title='Status')

plt.suptitle('Numerical Feature Distributions by Delivery Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("outputs/eda_numerical_distributions.png", dpi=150)
plt.show()
print("✅ Numerical distributions saved.")

# ─────────────────────────────────────────────
# 3. CATEGORICAL FEATURE ANALYSIS
# ─────────────────────────────────────────────
cat_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['Reached.on.Time_Y.N'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#ff6b6b', '#51cf66'], edgecolor='white')
    axes[i].set_title(f'{col} vs Delivery Status (%)', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Percentage')
    axes[i].set_xlabel(col)
    axes[i].legend(['Delayed', 'On Time'])
    axes[i].tick_params(axis='x', rotation=0)

plt.suptitle('Categorical Feature Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda_categorical_analysis.png", dpi=150)
plt.show()
print("✅ Categorical analysis saved.")

# ─────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────
# Encode for correlation
df_encoded = df.copy()
df_encoded['Mode_of_Shipment']   = df_encoded['Mode_of_Shipment'].map({'Ship': 0, 'Flight': 1, 'Road': 2})
df_encoded['Product_importance'] = df_encoded['Product_importance'].map({'Low': 0, 'Medium': 1, 'High': 2})
df_encoded['Gender']             = df_encoded['Gender'].map({'F': 0, 'M': 1})
df_encoded['Warehouse_block']    = df_encoded['Warehouse_block'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4})

if 'ID' in df_encoded.columns:
    df_encoded.drop(columns=['ID'], inplace=True)

plt.figure(figsize=(12, 8))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png", dpi=150)
plt.show()
print("✅ Correlation heatmap saved.")

# ─────────────────────────────────────────────
# 5. BOXPLOTS — OUTLIER DETECTION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(['Discount_offered', 'Weight_in_gms', 'Cost_of_the_Product']):
    sns.boxplot(data=df, x='Reached.on.Time_Y.N', y=col, palette='Set2', ax=axes[i])
    axes[i].set_title(f'{col} by Status', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('0=Delayed | 1=On Time')

plt.suptitle('Outlier Analysis — Key Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda_boxplots.png", dpi=150)
plt.show()
print("✅ Boxplots saved.")

print("\n🎉 EDA Complete! All charts saved in outputs/ folder.")
