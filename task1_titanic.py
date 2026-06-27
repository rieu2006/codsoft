# ============================================================
# TASK 1: TITANIC SURVIVAL PREDICTION
# CodSoft Data Science Internship
# Dataset: https://www.kaggle.com/datasets/brendan45774/test-file
# ============================================================

# ── STEP 1: IMPORT LIBRARIES ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

import warnings
warnings.filterwarnings('ignore')

# ── STEP 2: LOAD DATA ───────────────────────────────────────
# Place tested.csv (or train.csv) in the same folder as this script
df = pd.read_csv('tested.csv')

print("Shape:", df.shape)
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# ── STEP 3: EXPLORATORY DATA ANALYSIS ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Overall survival count
df['Survived'].value_counts().plot(kind='bar', ax=axes[0],
                                    color=['salmon', 'steelblue'])
axes[0].set_title('Survival Count (0=No, 1=Yes)')
axes[0].set_xlabel('Survived')
axes[0].set_ylabel('Count')

# Survival rate by gender
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[1],
                                           color=['pink', 'lightblue'])
axes[1].set_title('Survival Rate by Gender')
axes[1].set_ylabel('Survival Rate')

# Survival rate by passenger class
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[2],
                                              color='mediumpurple')
axes[2].set_title('Survival Rate by Pclass')
axes[2].set_ylabel('Survival Rate')

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=150)
plt.show()

# ── STEP 4: PREPROCESSING ──────────────────────────────────
# Fill missing Age with median (robust to outliers)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many nulls), Name, Ticket, PassengerId (irrelevant)
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Encode categorical columns to numbers
le = LabelEncoder()
df['Sex']      = le.fit_transform(df['Sex'])       # female=0, male=1
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

print("\nAfter Preprocessing:\n", df.head())

# ── STEP 5: FEATURE ENGINEERING ────────────────────────────
# Total family size on board
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Was the passenger travelling alone?
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Bin age into life stages: Child / Teen / Adult / Middle-age / Senior
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80],
                       labels=[0, 1, 2, 3, 4]).astype(int)

# Bin fare into 4 quartile groups
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)

print("\nFeatures After Engineering:\n", df.head())

# ── STEP 6: SPLIT DATA ──────────────────────────────────────
X = df.drop('Survived', axis=1)   # All columns except target
y = df['Survived']                 # Target: 0 = died, 1 = survived

# 80% train / 20% test, stratify keeps class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── STEP 7: TRAIN MODELS ────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds,
          target_names=['Did Not Survive', 'Survived']))

# ── STEP 8: CONFUSION MATRIX ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, model) in zip(axes, models.items()):
    cm   = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\nAccuracy: {results[name]:.2%}')

plt.tight_layout()
plt.savefig('titanic_confusion.png', dpi=150)
plt.show()

# ── STEP 9: FEATURE IMPORTANCE (Random Forest) ─────────────
rf = models['Random Forest']
feat_imp = pd.Series(rf.feature_importances_,
                     index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
feat_imp.plot(kind='bar', color='steelblue')
plt.title('Feature Importances – Random Forest')
plt.tight_layout()
plt.savefig('titanic_features.png', dpi=150)
plt.show()

# ── STEP 10: BEST MODEL ─────────────────────────────────────
print("\nBest model:", max(results, key=results.get),
      f"→ {max(results.values()):.2%}")
