import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTETomek

# 1. Load data
path = r'D:\BIM\Summer Project\project\ml_workspace\datasets\encoded_ibm_dataset.pkl'
data = pd.read_pickle(path)

X = data.drop('Attrition', axis=1)
y = data['Attrition']
feature_names = X.columns.tolist()
print("Original feature columns:", feature_names)

# 2. Handle imbalance with SMOTETomek
print("Original class distribution:", Counter(y))
smt = SMOTETomek(sampling_strategy=0.8, random_state=42)
X_res, y_res = smt.fit_resample(X, y)
print("Balanced class distribution:", Counter(y_res))

# 3. Train-test split (only after resampling)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
print("Train/Test split:", X_train.shape, X_test.shape)
print("Test label distribution:", Counter(y_test))

# 4. Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
encoder_path = r'D:\BIM\Summer Project\project\ml_workspace\saved_outputs\encoders.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump({'scaler': scaler, 'columns': feature_names}, f)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
predictions = model.predict(X_test_scaled)
print("Prediction distribution:", Counter(predictions))

accuracy = accuracy_score(y_test, predictions)
print(f"\nModel accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(r'D:\BIM\Summer Project\project\ml_workspace\saved_outputs\confusion_matrix.png')
plt.show()

# 8. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(6, 8))
plt.title("Feature Importance")
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.tight_layout()
plt.show()

# 9. Save model
model_path = r'D:\BIM\Summer Project\project\ml_workspace\saved_outputs\random-forest-model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump((model, feature_names), f)