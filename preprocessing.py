import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew

# ====================  1: Load Data ====================

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = train['SalePrice']

# ====================  2: Exploratory Data Analysis ====================

# SalePrice Distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, bins=40)
plt.title("SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("saleprice_distribution.png")
plt.close()

# Correlation Heatmap
corr = train.corr(numeric_only=True)
top_corr = corr['SalePrice'].abs().sort_values(ascending=False)[1:11]
plt.figure(figsize=(10, 6))
sns.heatmap(train[top_corr.index.tolist() + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Top Correlated Features with SalePrice")
plt.tight_layout()
plt.savefig("top_correlated_features.png")
plt.close()

# Missing Values Bar Chart
missing_vals = train.isnull().sum()
missing_vals = missing_vals[missing_vals > 0].sort_values()
plt.figure(figsize=(8, 4))
missing_vals.plot(kind='barh')
plt.title("Missing Values in Training Data")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("missing_values.png")
plt.close()

# ====================  3: Combine Data for Uniform Processing ====================

train['is_train'] = 1
test['is_train'] = 0
full = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)

# ====================  4: Handle Missing Values ====================

# Fill 'None' for features where NA means absence
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    full[col] = full[col].fillna('None')

# Median fill for numeric columns
num_impute_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                   'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                   'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in num_impute_cols:
    full[col] = full[col].fillna(full[col].median())

# Mode fill for categorical columns
mode_cols = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd',
             'SaleType', 'KitchenQual', 'Electrical']
for col in mode_cols:
    full[col] = full[col].fillna(full[col].mode()[0])

# Fill LotFrontage using median per neighborhood
full['LotFrontage'] = full.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Drop remaining columns with very few missing values if any
full.drop(columns=full.columns[full.isnull().any()], inplace=True)

# ====================  5: Feature Engineering ====================

full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']

full['TotalBath'] = (full['BsmtFullBath'] + 0.5 * full['BsmtHalfBath'] +
                     full['FullBath'] + 0.5 * full['HalfBath'])

full['HouseAge'] = full['YrSold'] - full['YearBuilt']
full['RemodelAge'] = full['YrSold'] - full['YearRemodAdd']
full['GarageAge'] = full['YrSold'] - full['GarageYrBlt']

# Drop original columns replaced by engineered ones
full.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# ====================  6: Fix Skewed Numeric Features ====================

numeric_feats = full.select_dtypes(include=[np.number]).columns
skewed_feats = full[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed = skewed_feats[abs(skewed_feats) > 0.75].index

for feat in skewed:
    full[feat] = np.log1p(full[feat])

# ====================  7: Encoding Categorical Features ====================

categorical_cols = full.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded = encoder.fit_transform(full[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=full.index)

# Combine with numeric
full.drop(categorical_cols, axis=1, inplace=True)
full_final = pd.concat([full, encoded_df], axis=1)

# ====================  8: Final Split and Save ====================

X_train = full_final[full['is_train'] == 1].drop('is_train', axis=1)
X_test = full_final[full['is_train'] == 0].drop('is_train', axis=1)

# Save files
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
y.to_csv("y_train.csv", index=False)

print("✅ Preprocessing and Feature Engineering complete.")

