import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------------
# 1. Load data & separate features / target
# ------------------------------------------------------------------
df = pd.read_csv('../../dataframe_grams_antibiotic.csv')

# Features: all numeric columns *except* the target column
X = df.select_dtypes(include='number').drop(columns=['event_status'], errors='ignore')
y_raw = df['event_status']          # <-- this is the string label

# ------------------------------------------------------------------
# 2. Encode the target (0 / 1)
# ------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y_raw)   # 0 for 'negatief', 1 for 'positief'

# ------------------------------------------------------------------
# 3. Drop rows that have missing numeric values (or impute if you prefer)
# ------------------------------------------------------------------
X = X.dropna()
y = y[X.index]          # keep the target aligned with dropped rows

# ------------------------------------------------------------------
# 4. Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=0,
    stratify=y            # keeps class balance in both sets
)

# ------------------------------------------------------------------
# 5. Standardise the features
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------------
# 6. PCA – keep two components (you can change n_components if you like)
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=0)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print(X.columns)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# ------------------------------------------------------------------
# 7. Logistic Regression
# ------------------------------------------------------------------
clf = LogisticRegression(random_state=0, max_iter=200)
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------
# 8. Plot decision boundary on the *training* set (PC1 vs PC2)
# ------------------------------------------------------------------
X_set, y_set = X_train_pca, y_train

# Create a meshgrid
x1_min, x1_max = X_set[:,0].min()-1, X_set[:,0].max()+1
x2_min, x2_max = X_set[:,1].min()-1, X_set[:,1].max()+1
xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, 0.01),
    np.arange(x2_min, x2_max, 0.01)
)

# Predict on the grid
Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

plt.figure(figsize=(12,10))
cmap_plot = ListedColormap(('yellow', 'white', 'aquamarine'))
plt.contourf(xx1, xx2, Z, alpha=0.75, cmap=cmap_plot)

# Scatter the training points
colors = ['red', 'green', 'blue']   # enough for 3 classes; adjust if needed
for idx, cls in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==cls, 0], X_set[y_set==cls, 1],
        color=colors[idx % len(colors)],
        label=le.classes_[cls]
    )

plt.title('Logistic Regression (Training set) – PCA space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

plt.figure(figsize=(8,4))
plt.bar(range(1, len(evr)+1), evr, alpha=0.6, label='Individual')
plt.step(range(1, len(cum_evr)+1), cum_evr, where='mid', color='red', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Explained variance ratio')
plt.title('Scree Plot')
plt.legend()
plt.show()
