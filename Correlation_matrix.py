import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: DATA PREPROCESSING & IMPUTATION (from your code)
# ============================================
df = pd.read_csv('../../dataframe_grams_antibiotic.csv', low_memory=False)

# Remove marked for deletion/contaminated samples
to_delete_ids = df.loc[df['delete'] == "Yes", 'SpecimenId_NL_CM_13_1_15'].unique()
to_delete_ids2 = df.loc[df['contaminated'] == "yes", 'SpecimenId_NL_CM_13_1_15'].unique()

df = df[~df['SpecimenId_NL_CM_13_1_15'].isin(to_delete_ids)].copy()
df = df[~df['SpecimenId_NL_CM_13_1_15'].isin(to_delete_ids2)].copy()

# Keep only relevant columns for analysis
df = df.drop(columns=['SpecimenId_NL_CM_13_1_15', 'delete', 'contaminated'], errors='ignore')
df_cat = df.select_dtypes(include='object').drop(columns=['event_status'], errors='ignore')
df_num = df.select_dtypes(include='number').drop(columns=['event_status'], errors='ignore')

# ============================================
# STEP 3: CREATE CORRELATION MATRIX
# ============================================

# OPTION 1: Correlation for NUMERIC VARIABLES ONLY
numeric_corr_matrix = df_num.corr()

# OPTION 2: Extended correlation including CATEGORICAL (using point-biserial/phi)
# This is more complex, requires converting categorical to numeric encoding first

def encode_categorical_for_correlation(df):
    """Encode categorical columns as numeric for correlation analysis"""
    df_encoded = df.copy()
    
    for col in df_cat.columns:
        # Use LabelEncoder (preserves order if ordinal) or OneHot for nominal
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 20:  # Only encode low-cardinality categories
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded_values = le.fit_transform(df[col].astype(str))
            df_encoded[f'{col}_numeric'] = encoded_values
        else:
            # Skip high-cardinality for correlation matrix
            continue
    
    return df_encoded

df_for_corr = encode_categorical_for_correlation(df)

# Combine numeric imputed + categorical encoded
corr_features = list(df_num.columns) + [c + '_numeric' for c in df_cat.columns if c + '_numeric' in df_for_corr.columns]
X_combined_for_corr = df_for_corr[corr_features].dropna()

# Calculate correlation matrix (Pearson for numeric, can use Spearman for robustness)
correlation_matrix = X_combined_for_corr.corr(method='pearson')  # Use 'spearman' if ordinal data exists

print("\nCorrelation Matrix Shape:", correlation_matrix.shape)
print(f"Total features in correlation matrix: {len(corr_features)}")

# ============================================
# STEP 4: VISUALIZE CORRELATION MATRIX (HEATMAP)
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# LEFT PLOT: Correlation Heatmap
sns.heatmap(correlation_matrix, 
           annot=False,          # Set to True if you want numbers
           fmt='.2f',            # Decimal places (for annotations)
           cmap='RdBu_r',        # Red-blue colormap for positive/negative
           linewidths=0.5,       # Line width between cells
           linecolor='gray',     # Color of grid lines
           ax=axes[0])

axes[0].set_title('Correlation Matrix (Pearson Correlation)', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Features')

axes[0].set_yticks(range(len(correlation_matrix)))
axes[0].set_xticks(range(len(correlation_matrix)))
axes[0].set_yticklabels([col.replace('_numeric', '').replace('Gram_', '') 
                          for col in correlation_matrix.columns])
axes[0].set_xticklabels([col.replace('_numeric', '').replace('Gram_', '') 
                         for col in correlation_matrix.columns], rotation=45, ha='right')



# Actually simpler: use seaborn matrix with color threshold
axes[1].set_title('Correlation Heatmap (Thresholded)', 
                  fontsize=14, fontweight='bold')

sns.heatmap(correlation_matrix,
           annot=False,
           cmap='RdBu_r',
           vmin=-0.5, vmax=0.5,
           center=0,
           linewidths=0.5,
           linecolor='gray',
           ax=axes[1],
           cbar_kws={"shrink": 0.8})

plt.tight_layout()
plt.show()