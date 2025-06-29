import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel('dataset/HealthCareData.xlsx', engine='openpyxl')
df.dropna(inplace=True)

# Rename target column
df.rename(columns={
    'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)': 'Target'
}, inplace=True)

# Keep only the required 10 features + target
columns_to_use = [
    'Age',
    'Gender',
    'Duration of alcohol consumption(years)',
    'Quantity of alcohol consumption (quarters/day)',
    'Hepatitis B infection',
    'Diabetes Result',
    'Obesity',
    'Hemoglobin  (g/dl)',
    'Platelet Count  (lakhs/mm)',
    'USG Abdomen (diffuse liver or  not)',
    'Target'
]
df = df[columns_to_use]

# Encode categorical values
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split features/labels
X = df.drop(columns=['Target'])
y = df['Target']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model and scaler
pickle.dump(model, open('models/rf_acc_68.pkl', 'wb'))
pickle.dump(scaler, open('models/normalizer.pkl', 'wb'))

print("✅ Model and Scaler saved with 10 features")
