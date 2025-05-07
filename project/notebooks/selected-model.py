# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.preprocessing import StandardScaler  
from imblearn.over_sampling import RandomOverSampler  

import pickle

# Load the dataset
file_path = 'D:\\BIM\\Summer Project\\datasets\\cleaned_ibm_dataset.pkl'
df = pd.read_pickle(file_path)

# Split into features (X) and target variable (y)
X = df.drop('Attrition', axis=1)  
y = df['Attrition']               

# Standardize the data before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame to retain column names
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Handle class imbalance using oversampling (Only on training data)
print(f"Before oversampling: {Counter(y_train)}")
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)
print(f"After oversampling: {Counter(y_train)}")

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='g', cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
# plt.show()

# Feature Importance Plot
feature_names = df.drop('Attrition', axis=1).columns  
feature_importance = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 8))
plt.title("Feature Importance")
plt.barh(range(len(feature_names)), feature_importance[sorted_indices])
plt.yticks(range(len(feature_names)), labels=[feature_names[i] for i in sorted_indices])
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
# plt.show()

# Save the model
model_path = 'D:\\BIM\\Summer Project\\models\\random-forest-model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf_model, file)
