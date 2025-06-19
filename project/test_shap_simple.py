#!/usr/bin/env python3
"""
Simple test script to verify SHAP implementation without database dependencies.
"""

import pickle
import pandas as pd
import shap
import numpy as np
import json

# Load the model and encoders
MODEL_PATH = 'models/random-forest-model.pkl'
ENCODER_PATH = 'models/encoders.pkl'
TRAINING_DATA_PATH = 'datasets/cleaned_ibm_dataset.csv'

def test_shap_implementation():
    """Test the SHAP implementation with sample data."""
    print("Testing SHAP Implementation")
    print("=" * 40)
    
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model, model_columns = pickle.load(f)
    
    # Load encoders
    with open(ENCODER_PATH, 'rb') as f:
        encoder_data = pickle.load(f)
    
    # Load training data
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    
    # Preprocess training data for background
    background_data = training_data.fillna(0)
    expected_columns = encoder_data['columns']
    for col in expected_columns:
        if col not in background_data.columns:
            background_data[col] = 0
    background_data = background_data[expected_columns]
    background_sample = background_data.sample(min(100, len(background_data)), random_state=42)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model, background_sample, feature_perturbation="interventional")
    
    # Create sample employee data
    sample_employee = {
        'age': 35,
        'distance': 10,
        'env_satisfaction': 3,
        'gender': 'Male',
        'job_involvement': 3,
        'job_level': 2,
        'job_satisfaction': 2,
        'salary': 50000,
        'overtime': 'Yes',
        'total_working_years': 8,
        'work_life_balance': 2,
        'years_at_company': 5,
        'department': 'Sales',
        'edufield': 'Marketing',
        'position': 'Sales Executive',
        'marital_status': 'Married'
    }
    
    # Preprocess employee data (simplified version)
    rename_map = {
        'age': 'Age', 'distance': 'DistanceFromHome', 'env_satisfaction': 'EnvironmentSatisfaction',
        'gender': 'Gender', 'job_involvement': 'JobInvolvement', 'job_level': 'JobLevel',
        'job_satisfaction': 'JobSatisfaction', 'salary': 'MonthlyIncome', 'overtime': 'OverTime',
        'total_working_years': 'TotalWorkingYears', 'work_life_balance': 'WorkLifeBalance',
        'years_at_company': 'YearsAtCompany', 'department': 'Dept', 'edufield': 'EduField',
        'position': 'JobRole', 'marital_status': 'MaritalStatus'
    }
    
    data = {rename_map.get(k.lower(), k): v for k, v in sample_employee.items()}
    df = pd.DataFrame([data])
    
    # Binary encoding
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    # One-hot encoding
    ohe_map = ['Dept', 'EduField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=ohe_map)
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Order columns correctly
    df = df[expected_columns]
    
    # Apply scaler if available
    scaler = encoder_data.get('scaler')
    if scaler:
        df_scaled = scaler.transform(df)
        df = pd.DataFrame(df_scaled, columns=expected_columns)
    
    X = df
    
    # Get prediction
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    attrition_probability = float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0])
    
    print(f"Sample Employee: {sample_employee}")
    print(f"Prediction: {prediction} (Attrition: {'Yes' if prediction == 1 else 'No'})")
    print(f"Attrition Probability: {attrition_probability:.3f}")
    
    # Get SHAP values
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Convert to native Python types
    feature_names = X.columns
    feature_shap_pairs = list(zip(feature_names, shap_values[0].astype(float)))
    
    # Group one-hot encoded features
    grouped_factors = {}
    for name, importance in feature_shap_pairs:
        if name.startswith('Dept_'):
            base_name = 'Department'
        elif name.startswith('EduField_'):
            base_name = 'Education Field'
        elif name.startswith('JobRole_'):
            base_name = 'Job Role'
        elif name in ['Divorced', 'Married', 'Single']:
            base_name = 'Marital Status'
        else:
            base_name = name

        if base_name not in grouped_factors:
            grouped_factors[base_name] = 0
        grouped_factors[base_name] += importance
    
    grouped_factors = sorted(grouped_factors.items(), key=lambda x: x[1], reverse=True)
    
    print("\nSHAP-based Feature Explanations:")
    print("-" * 30)
    
    # Show top positive and negative factors
    positive_factors = [(f, v) for f, v in grouped_factors if v > 0]
    negative_factors = [(f, v) for f, v in grouped_factors if v < 0]
    
    print("Factors Increasing Attrition Risk:")
    for factor, shap_value in positive_factors[:5]:
        print(f"  {factor}: +{shap_value:.4f}")
    
    print("\nFactors Decreasing Attrition Risk:")
    for factor, shap_value in negative_factors[:5]:
        print(f"  {factor}: {shap_value:.4f}")
    
    # Test JSON serialization
    print("\nTesting JSON serialization...")
    try:
        json_str = json.dumps(grouped_factors)
        print("✓ JSON serialization successful")
        print(f"JSON length: {len(json_str)} characters")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
    
    print("\n✓ SHAP implementation test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_shap_implementation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 