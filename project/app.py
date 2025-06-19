from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash, abort
import random # For simulating data, remove in production
import mysql.connector
from datetime import datetime, timedelta, date
import bcrypt  # For password hashing
from functools import wraps
from flask_mail import Mail, Message
import config  # Import the config module
import pickle
import pandas as pd
from calendar import monthrange
from markupsafe import Markup
import re
import json
from collections import defaultdict
import shap
import numpy as np

app = Flask(__name__)

app.secret_key = config.SECRET_KEY # Use the secret key from config

app.config.update(config.MAIL_CONFIG) # Configure email settings using config

mail = Mail(app)

MODEL_PATH = 'D:/BIM/Summer Project/project/models/random-forest-model.pkl'
ENCODER_PATH = 'D:/BIM/Summer Project/project/models/encoders.pkl'
TRAINING_DATA_PATH = 'D:/BIM/Summer Project/project/datasets/cleaned_ibm_dataset.csv'

# Load the model and encoders
with open(MODEL_PATH, 'rb') as f:
    model, model_columns = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    encoder_data = pickle.load(f)
    model_columns = encoder_data['columns']

# Load training data for SHAP background dataset
def load_training_data():
    """Load and preprocess training data for SHAP background dataset."""
    try:
        # Load the training dataset
        training_data = pd.read_csv(TRAINING_DATA_PATH)
        
        # Apply the same preprocessing as used during training
        # This should match the preprocessing in your training script
        processed_data = preprocess_training_data(training_data)
        
        # Use a sample of the data as background (SHAP works well with 100-1000 samples)
        background_data = processed_data.sample(min(500, len(processed_data)), random_state=42)
        
        return background_data
    except Exception as e:
        print(f"Warning: Could not load training data for SHAP background: {e}")
        return None

def preprocess_training_data(df):
    """Preprocess training data to match the model input format."""
    # This should match your training preprocessing
    # For now, using a simplified version - you may need to adjust based on your actual training preprocessing
    
    # Handle missing values
    df = df.fillna(0)
    
    # Ensure all expected columns are present
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the columns expected by the model
    df = df[model_columns]
    
    return df

# Initialize SHAP explainer
background_data = load_training_data()
if background_data is not None:
    # Create TreeExplainer with background data for more accurate SHAP values
    explainer = shap.TreeExplainer(model, background_data, feature_perturbation="interventional")
else:
    # Fallback to tree_path_dependent if no background data
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

# Leave day limits
LEAVE_LIMITS = {
    'Annual Leave': 5,
    'Sick Leave': 5,
    'Personal Leave': 5
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or session.get('role') != role:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def get_db_connection():
    # Use the database configuration from config
    connection = mysql.connector.connect(**config.DB_CONFIG)
    return connection

def calculate_leave_days(start_date, end_date, duration):
    """Calculate the total number of leave days."""
    # Convert date strings to datetime objects
    start_date_obj = datetime.strptime(str(start_date), '%Y-%m-%d')
    end_date_obj = datetime.strptime(str(end_date), '%Y-%m-%d')
    
    # Calculate total days
    total_days = (end_date_obj - start_date_obj).days + 1
    
    # For half day leave
    if duration == 'Half Day':
        total_days = total_days / 2
    
    return total_days

def calculate_age(dob):
    """Calculate age from date of birth (dob: 'YYYY-MM-DD' or date object)."""
    if not dob:
        return ''
    if isinstance(dob, str):
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d')
        except Exception:
            return ''
    else:
        dob_date = dob
    today = datetime.today()
    return today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))

def get_leave_requests():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT lr.*, e.name, e.department as dept 
        FROM leave_requests lr
        JOIN employees e ON lr.employee_id = e.employee_id
        ORDER BY lr.start_date DESC
    """)
    leave_requests = cursor.fetchall()
    
    # Calculate total days for each leave request
    for request in leave_requests:
        request['total_days'] = calculate_leave_days(
            request['start_date'], request['end_date'], request['duration']
        )

    return leave_requests

def get_all_employees():
    """Fetch all employees from the database."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    cursor.close()
    connection.close()
    return employees

def get_employee_by_id(employee_id):
    """Fetch an employee by ID from the database."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE employee_id = %s", (employee_id,))
    employee = cursor.fetchone()
    cursor.close()
    connection.close()
    return employee

def preprocess_employee_data(raw_data):
    """
    Preprocess raw employee data from the database to match model input format.
    Applies encoding and scaling using the saved encoder_data.

    Parameters:
        raw_data (dict): A single employee's data as a dictionary.

    Returns:
        pd.DataFrame: Preprocessed input ready for prediction.
    """
    # Load expected model structure
    expected_columns = encoder_data['columns']
    scaler = encoder_data.get('scaler')

    # Rename DB fields to match model training columns
    rename_map = {
        'age': 'Age',
        'distance': 'DistanceFromHome',
        'env_satisfaction': 'EnvironmentSatisfaction',
        'gender': 'Gender',
        'job_involvement': 'JobInvolvement',
        'job_level': 'JobLevel',
        'job_satisfaction': 'JobSatisfaction',
        'salary': 'MonthlyIncome',
        'overtime': 'OverTime',
        'total_working_years': 'TotalWorkingYears',
        'work_life_balance': 'WorkLifeBalance',
        'years_at_company': 'YearsAtCompany',
        'department': 'Dept',
        'edufield': 'EduField',
        'position': 'JobRole',
        'marital_status': 'MaritalStatus'
    }

    # Step 1: Normalize input and rename fields
    data = {rename_map.get(k.lower(), k): v for k, v in raw_data.items()}
    df = pd.DataFrame([data])

    # Step 2: Binary encoding
    binary_map = {
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0}
    }
    for col, mapping in binary_map.items():
        if col in df:
            df[col] = df[col].map(mapping).fillna(0)
        else:
            df[col] = 0

    # Step 3: Handle categorical columns (OHE)
    ohe_map = ['Dept', 'EduField', 'JobRole', 'MaritalStatus']
    defaults = {
        'Dept': 'Sales',
        'EduField': 'Medical',
        'JobRole': 'Sales Executive',
        'MaritalStatus': 'Single'
    }
    for col in ohe_map:
        if col not in df.columns:
            df[col] = defaults[col]

    df = pd.get_dummies(df, columns=ohe_map)

    # Step 4: Ensure all expected model columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Step 5: Order columns correctly
    df = df[expected_columns]

    # Step 6: Apply scaler if available
    if scaler:
        df_scaled = scaler.transform(df)
        df = pd.DataFrame(df_scaled, columns=expected_columns)

    return df

def group_one_hot_features(feature_names, feature_importance):
    """Group one-hot encoded features and sum their importance values.
    
    Args:
        feature_names: List of feature names from the model
        feature_importance: List of importance values from the model
        
    Returns:
        List of tuples (feature_name, importance) sorted by importance
    """
    grouped_factors = {}
    for name, importance in zip(feature_names, feature_importance):
        # Handle one-hot encoded features
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

    # Convert to sorted list
    return sorted(grouped_factors.items(), key=lambda x: x[1], reverse=True)

def get_shap_explanations(X):
    """Get SHAP values for a single employee's data.
    
    Args:
        X: Preprocessed employee data as DataFrame
        
    Returns:
        List of tuples (feature_name, shap_value) sorted by absolute SHAP value
    """
    try:
        # Get SHAP values for this specific prediction
        shap_values = explainer.shap_values(X)
        
        # For binary classification, get the values for class 1 (attrition)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get feature names and their SHAP values
        feature_names = X.columns
        
        # Create list of (feature_name, shap_value) tuples
        feature_shap_pairs = list(zip(feature_names, shap_values[0].astype(float)))
        
        # Group one-hot encoded features and their SHAP values
        grouped_factors = {}
        for name, value in feature_shap_pairs:
            # Handle one-hot encoded features
            if name.startswith('Dept_'):
                base_name = 'Department: ' + name.replace('Dept_', '')
            elif name.startswith('EduField_'):
                base_name = 'Education: ' + name.replace('EduField_', '')
            elif name.startswith('JobRole_'):
                base_name = 'Position: ' + name.replace('JobRole_', '')
            elif name in ['Divorced', 'Married', 'Single']:
                base_name = 'Status: ' + name
            else:
                # Clean up other feature names
                base_name = name.replace('_', ' ').title()
            
            grouped_factors[base_name] = value

        # Sort by absolute value for importance
        sorted_factors = sorted(
            grouped_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_factors
        
    except Exception as e:
        print(f"Error getting SHAP explanations: {e}")
        return []

def predict_attrition_for_employee(employee_id):
    emp = get_employee_by_id(employee_id)
    if not emp:
        return
    X = preprocess_employee_data(emp)
    prediction = int(model.predict(X)[0])
    
    # Get SHAP-based feature explanations for this specific employee
    all_factors = get_shap_explanations(X)
    
    # Also get prediction probability for more detailed insights
    prediction_proba = model.predict_proba(X)[0]
    attrition_probability = float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0])
    
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE employees
        SET attrition_risk = %s,
            attrition_factors = %s,
            attrition_probability = %s
        WHERE employee_id = %s
    """, (prediction, json.dumps(all_factors), attrition_probability, employee_id))
    connection.commit()
    cursor.close()
    connection.close()

def predict_attrition_for_all():
    """Predict attrition for all employees and update the database."""
    employees = get_all_employees()
    connection = get_db_connection()
    cursor = connection.cursor()
    for emp in employees:
        X = preprocess_employee_data(emp)
        prediction = int(model.predict(X)[0])
        
        # Get SHAP-based feature explanations for this specific employee
        all_factors = get_shap_explanations(X)
        
        # Also get prediction probability for more detailed insights
        prediction_proba = model.predict_proba(X)[0]
        attrition_probability = float(prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0])
        
        cursor.execute("""
            UPDATE employees
            SET attrition_risk = %s,
                attrition_factors = %s,
                attrition_probability = %s
            WHERE employee_id = %s
        """, (prediction, json.dumps(all_factors), attrition_probability, emp['employee_id']))
    connection.commit()
    cursor.close()
    connection.close()

def get_remaining_leave_days(employee_id, leave_type):
    """Calculate remaining leave days for a specific type of leave."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        # Get total leave days used for this type
        cursor.execute("""
            SELECT SUM(
                CASE 
                    WHEN duration = 'Half Day' THEN 0.5
                    ELSE 1
                END * (DATEDIFF(end_date, start_date) + 1)
            ) as total_days
            FROM leave_requests 
            WHERE employee_id = %s 
            AND type = %s 
            AND status = 'Approved'
            AND YEAR(start_date) = YEAR(CURRENT_DATE)
        """, (employee_id, leave_type))
        result = cursor.fetchone()
        used_days = result['total_days'] if result['total_days'] is not None else 0
        
        # Calculate remaining days
        remaining_days = LEAVE_LIMITS.get(leave_type, 0) - used_days
        return max(0, remaining_days)
    finally:
        cursor.close()
        connection.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Fetch user by email
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        # Check if password is bcrypt hash or plaintext
        db_password = user['password']
        if db_password.startswith('$2'):
            # bcrypt hash
            valid = bcrypt.checkpw(password.encode('utf-8'), db_password.encode('utf-8'))
        else:
            # plaintext fallback (should not happen after migration)
            valid = password == db_password
        if not valid:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        session['user_id'] = user['id']
        session['role'] = user['role']
        session['employee_id'] = user['employee_id']
        session['email'] = user['email']
        
        # Fetch employee details
        cursor.execute("SELECT name, position FROM employees WHERE employee_id = %s", (user['employee_id'],))
        employee = cursor.fetchone()
        if not employee:
            flash('Employee record not found for this user.', 'error')
            cursor.close()
            connection.close()
            return redirect(url_for('login'))
        session['name'] = employee['name']
        session['position'] = employee['position']
        
        if user['role'] == 'hr':
            cursor.close()
            connection.close()
            return redirect(url_for('dashboard'))
        elif user['role'] == 'employee':
            cursor.close()
            connection.close()
            return redirect(url_for('employee_view'))
        else:
            flash('Invalid user role', 'error')
            cursor.close()
            connection.close()
            return redirect(url_for('login'))

        cursor.close()
        connection.close()
    else:
        return render_template('auth/login.html')

def get_departments():
    """Fetch unique department names from the employees table."""
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT department FROM employees")
    departments = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return departments

@app.route('/job_apply', methods=['GET', 'POST'])
def job_apply():
    today = date.today()
    min_dob = (today.replace(year=today.year - 60)).isoformat()
    max_dob = (today.replace(year=today.year - 18)).isoformat()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        dob = request.form['dob']
        position = request.form['position']
        department = request.form['department']
        edufield = request.form['education_field']
        total_working_years = request.form['total_working_years']
        overtime = request.form['overtime']
        distance = request.form['distance_from_home']
        marital_status = request.form['marital_status']
        gender = request.form['gender']

        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Please enter a valid email address.', 'error')
            return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

        # Validate total_working_years input
        try:
            total_working_years = int(total_working_years)
            if total_working_years < 0:
                flash('Total working years cannot be negative.', 'error')
                return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)
        except ValueError:
            flash('Total working years must be a valid number.', 'error')
            return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

        # Validate distance from home
        try:
            distance = int(distance)
            if distance < 0:
                flash('Distance from home cannot be negative.', 'error')
                return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)
        except ValueError:
            flash('Distance from home must be a valid number.', 'error')
            return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

        # Validate date of birth
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d')
            age = calculate_age(dob)
            if age < 18:
                flash('You must be at least 18 years old to apply.', 'error')
                return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)
            if age > 60:
                flash('You must be under 60 years old to apply.', 'error')
                return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)
        except ValueError:
            flash('Invalid date of birth format.', 'error')
            return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

        # Validate working years against age
        max_working_years = max(age - 18, 0)
        if total_working_years > max_working_years:
            flash(f'Total working years ({total_working_years}) cannot exceed your potential working years ({max_working_years}) based on your age ({age}).', 'error')
            return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

        # Connect to your database
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            sql = """
            INSERT INTO applicants
            (name, email, dob, position, department, edufield, total_working_years, overtime, distance, marital_status, gender)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                name, email, dob, position, department, edufield,
                total_working_years, overtime, int(distance),
                marital_status, gender
            ))
            conn.commit()
            cursor.close()
        finally:
            conn.close()
        flash('Application submitted successfully!', 'success')
        return redirect(url_for('job_apply'))
    return render_template('job_apply.html', min_dob=min_dob, max_dob=max_dob)

@app.route('/dashboard')
@login_required
@role_required('hr')
def dashboard():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch employee data from the database
    cursor.execute("SELECT COUNT(*) AS total FROM employees")
    total_employees = cursor.fetchone()['total']
    
    # Calculate attrition rate
    cursor.execute("SELECT COUNT(*) AS attrition_count FROM employees WHERE attrition_risk = 1")
    attrition_count = cursor.fetchone()['attrition_count']
    attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0

    # Fetch leave requests data from the database
    leave_requests = get_leave_requests()

    # Calculate at risk employees
    predict_attrition_for_all()  # Ensure attrition risk is updated
    cursor.execute("SELECT COUNT(*) AS at_risk_count FROM employees WHERE attrition_risk = 1")
    at_risk_count = cursor.fetchone()['at_risk_count']

    # Fetch employee data from the database
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()

    # --- Anniversaries logic (moved to function) ---
    anniversaries = get_anniversaries_this_month(employees)
    birthdays = get_birthdays_this_month(employees)

    # Get department distribution dynamically
    departments = get_departments()
    format_strings = ','.join(['%s'] * len(departments))
    cursor.execute(f"""
        SELECT department, COUNT(*) as count
        FROM employees
        WHERE department IN ({format_strings})
        GROUP BY department
    """, tuple(departments))
    dept_data = cursor.fetchall()
    department_labels = [row['department'] for row in dept_data]
    department_data = [row['count'] for row in dept_data]

    # Get age distribution
    cursor.execute("""
        SELECT 
            CASE 
                WHEN age < 25 THEN '18-24'
                WHEN age < 35 THEN '25-34'
                WHEN age < 45 THEN '35-44'
                WHEN age < 55 THEN '45-54'
                ELSE '55+'
            END as age_group,
            COUNT(*) as count
        FROM employees
        GROUP BY age_group
        ORDER BY MIN(age)
    """)
    age_data = cursor.fetchall()
    age_labels = [row['age_group'] for row in age_data]
    age_data_counts = [row['count'] for row in age_data]

    # Get salary distribution (5k bands)
    cursor.execute("""
        SELECT 
            CONCAT(FLOOR(salary/5000)*5, 'k-', FLOOR(salary/5000)*5+5, 'k') as salary_band,
            COUNT(*) as count
        FROM employees
        GROUP BY salary_band
        ORDER BY MIN(salary)
    """)
    salary_data_rows = cursor.fetchall()
    salary_labels = [row['salary_band'] for row in salary_data_rows]
    salary_data = [row['count'] for row in salary_data_rows]

    # Get leave type distribution
    cursor.execute("""
        SELECT type, COUNT(*) as count
        FROM leave_requests
        GROUP BY type
    """)
    leave_type_data = cursor.fetchall()
    leave_type_labels = [row['type'] for row in leave_type_data]
    leave_type_data = [row['count'] for row in leave_type_data]

    # Attrition by Gender
    cursor.execute("""
        SELECT gender, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY gender
    """)
    attrition_gender = cursor.fetchall()
    attrition_gender_labels = [row['gender'] or 'Unknown' for row in attrition_gender]
    attrition_gender_data = [row['count'] for row in attrition_gender]

    # Attrition by Age Group
   
    attrition_age_data = [row['count'] for row in attrition_age]

    # Attrition by Department
    cursor.execute(f"""
        SELECT department, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1 AND department IN ({format_strings})
        GROUP BY department
    """, tuple(departments))
    attrition_dept = cursor.fetchall()
    attrition_dept_labels = [row['department'] for row in attrition_dept]
    attrition_dept_data = [row['count'] for row in attrition_dept]

    # Attrition by Salary Band (5k bands)
    cursor.execute("""
        SELECT 
            CONCAT(FLOOR(salary/5000)*5, 'k-', FLOOR(salary/5000)*5+5, 'k') as salary_band,
            COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY salary_band
        ORDER BY MIN(salary)
    """)
    attrition_salary = cursor.fetchall()
    attrition_salary_labels = [row['salary_band'] for row in attrition_salary]
    attrition_salary_data = [row['count'] for row in attrition_salary]

    # Attrition by Overtime
    cursor.execute("""
        SELECT overtime, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY overtime
    """)
    attrition_overtime = cursor.fetchall()
    attrition_overtime_labels = [row['overtime'] or 'Unknown' for row in attrition_overtime]
    attrition_overtime_data = [row['count'] for row in attrition_overtime]

    # ...existing code...
    return render_template('hr/dashboard.html',
                        attrition_rate=attrition_rate,
                        total_employees=total_employees,
                        name=session.get('name'),
                        position=session.get('position'),
                        leave_requests=leave_requests,
                        employees=employees,
                        anniversaries=anniversaries,
                        birthdays=birthdays,
                        department_labels=department_labels,
                        department_data=department_data,
                        age_labels=age_labels,
                        age_data=age_data_counts,
                        salary_labels=salary_labels,
                        salary_data=salary_data,
                        at_risk_count=at_risk_count,
                        attrition_gender_labels=attrition_gender_labels,
                        attrition_gender_data=attrition_gender_data,
                        attrition_age_labels=attrition_age_labels,
                        attrition_age_data=attrition_age_data,
                        attrition_dept_labels=attrition_dept_labels,
                        attrition_dept_data=attrition_dept_data,
                        attrition_salary_labels=attrition_salary_labels,
                        attrition_salary_data=attrition_salary_data,
                        attrition_overtime_labels=attrition_overtime_labels,
                        attrition_overtime_data=attrition_overtime_data
                        )

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    session.pop('name', None)
    session.pop('position', None)
    session.pop('employee_id', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/employees')
@login_required
@role_required('hr')
def employees():
    employees = get_all_employees()
    return render_template("hr/employees.html", employees=employees)

@app.route('/employees/add_employee', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        position = request.form['position']
        department = request.form['department']
        salary = request.form['salary']
        date_joined = request.form['date_joined']

        # Insert the employee data into the database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Convert date_joined string to datetime object
        date_joined_obj = datetime.strptime(date_joined, '%Y-%m-%d')
        current_date = datetime.now()
        
        # Calculate years with decimal precision
        years_at_company = current_date.year - date_joined_obj.year

        # Get the next employee_id manually
        cursor.execute("SELECT MAX(employee_id) FROM employees")
        last_id = cursor.fetchone()[0] or 0
        employee_id = last_id + 1

        # Insert default values for other fields

        cursor.execute(
            "INSERT INTO employees (employee_id, name, position, department, salary, years_at_company) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (employee_id, name, position, department, salary, years_at_company)
        )
        # --- Create user account for the new employee ---
        # Generate a unique email for the employee
        base_email = f"{name.lower().replace(' ', '')}@company.com"
        email = base_email
        suffix = 1
        cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
        while cursor.fetchone()[0] > 0:
            email = f"{name.lower().replace(' ', '')}{suffix}@company.com"
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            suffix += 1
        raw_password = "default123"
        hashed_password = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        role = 'employee'
        cursor.execute(
            "INSERT INTO users (employee_id, email, password, role) VALUES (%s, %s, %s, %s)",
            (employee_id, email, hashed_password, role)
        )
        # --- End user account creation ---

        connection.commit()
        cursor.close()
        connection.close()

        # Predict attrition for the new employee
        predict_attrition_for_employee(employee_id)
        flash_attrition_status(employee_id)
        flash('Employee added successfully', 'success')
        return redirect(url_for('employees'))
    return render_template('hr/add_employee.html', employees=[])

@app.route('/employees/edit_employee/<int:employee_id>', methods=['GET', 'POST'])
def edit_employee(employee_id):
    if request.method == 'POST':
        name = request.form['name']
        position = request.form['position']
        department = request.form['department']
        salary = request.form['salary']
        date_joined = request.form['date_joined']
        # Validate salary is not negative
        try:
            salary_value = float(salary)
            if salary_value < 0:
                flash('Salary cannot be negative.', 'error')
                employee = get_employee_by_id(employee_id)
                return render_template('hr/edit_employee.html', employee=employee, employees=[])
        except ValueError:
            flash('Salary must be a valid number.', 'error')
            employee = get_employee_by_id(employee_id)
            return render_template('hr/edit_employee.html', employee=employee, employees=[])

        connection = get_db_connection()
        cursor = connection.cursor()

        # Convert date_joined string to datetime object
        date_joined_obj = datetime.strptime(date_joined, '%Y-%m-%d')
        current_date = datetime.now()
        # Calculate years with decimal precision
        years_at_company = current_date.year - date_joined_obj.year
        
        cursor.execute("""
            UPDATE employees
            SET name = %s, position = %s, department = %s, salary = %s
            , years_at_company = %s 
            WHERE employee_id = %s 
        """, (name, position, department, salary, years_at_company, employee_id))

        connection.commit()
        cursor.close()
        connection.close()

        # Predict attrition for the updated employee
        predict_attrition_for_employee(employee_id)
        # flash_attrition_status(employee_id)
        flash('Employee updated successfully', 'success')
        return redirect(url_for('employees'))
    
    employee = get_employee_by_id(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))
        
    return render_template('hr/edit_employee.html', employee=employee, employees=[])
        
@app.route('/employees/delete_employee/<int:employee_id>') 
def delete_employee(employee_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    # Delete from users table first to avoid foreign key constraint issues
    cursor.execute("DELETE FROM users WHERE employee_id = %s", (employee_id,))
    # Then delete from employees table
    cursor.execute("DELETE FROM employees WHERE employee_id = %s", (employee_id,))
    connection.commit()
    cursor.close()
    connection.close()

    flash('Employee deleted successfully', 'success')
    return redirect(url_for('employees'))

@app.route('/leave-requests')
def leave_requests():
    leave_requests = get_leave_requests()
    return render_template("hr/leave_requests.html",
                           leave_requests=leave_requests,
                           employees=[]
                           )

def send_hire_email(applicant_email, applicant_name, company_email, password):
    msg = Message(
        subject='Congratulations! You have been hired',
        recipients=[applicant_email]
    )
    msg.body = (
        f"Dear {applicant_name},\n\n"
        "Congratulations! You have been hired. You can now access your employee portal with the following credentials:\n\n"
        f"Login Email: {company_email}\n"
        f"Password: {password}\n\n"
        "Please log in and update your profile as soon as possible.\n\n"
        "Best regards,\nHR Team"
    )
    mail.send(msg)

@app.route('/applicant_action', methods=['POST'])
@login_required
@role_required('hr')
def applicant_action():
    reg_id = request.form.get('reg_id')
    action = request.form.get('action')
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    # Fetch applicant info
    cursor.execute("SELECT * FROM applicants WHERE reg_id = %s", (reg_id,))
    applicant = cursor.fetchone()
    if not applicant:
        flash('Applicant not found.', 'error')
        cursor.close()
        connection.close()
        return redirect(url_for('applicant_tracker'))

    if action == 'hire':
        salary = request.form.get('salary')
        date_joined = request.form.get('date_joined')
        # Calculate age using calculate_age()
        dob = applicant.get('dob')
        age = calculate_age(dob)
        # Calculate years at company from date_joined
        date_joined_obj = datetime.strptime(date_joined, '%Y-%m-%d')
        today = datetime.today()
        years_at_company = today.year - date_joined_obj.year - ((today.month, today.day) < (date_joined_obj.month, date_joined_obj.day))

        # Insert into employees with age, years_at_company, salary
        # Set default value 4 for env_satisfaction, job_involvement, job_satisfaction, work_life_balance
        cursor.execute("SELECT MAX(employee_id) FROM employees")
        last_id = cursor.fetchone()['MAX(employee_id)'] or 0
        employee_id = last_id + 1
        cursor.execute(
            "INSERT INTO employees (employee_id, name, position, department, salary, date_joined, edufield, total_working_years, overtime, distance, marital_status, gender, dob, age, years_at_company, env_satisfaction, job_involvement, job_satisfaction, work_life_balance) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                employee_id,
                applicant['name'],
                applicant['position'],
                applicant['department'],
                salary,
                date_joined,
                applicant.get('edufield', ''),
                applicant.get('total_working_years', 0),
                applicant.get('overtime', 'No'),
                applicant.get('distance', 0),
                applicant.get('marital_status', ''),
                applicant.get('gender', ''),
                applicant.get('dob', None),
                age,
                years_at_company,
                4,  # env_satisfaction
                4,  # job_involvement
                4,  # job_satisfaction
                4   # work_life_balance
            )
        )
        # Create user account
        name = applicant['name']
        base_email = f"{name.lower().replace(' ', '')}@company.com"
        email = base_email
        suffix = 1
        cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
        while cursor.fetchone()['COUNT(*)'] > 0:
            email = f"{name.lower().replace(' ', '')}{suffix}@company.com"
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            suffix += 1
        raw_password = "default123"
        hashed_password = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        role = 'employee'
        cursor.execute(
            "INSERT INTO users (employee_id, email, password, role) VALUES (%s, %s, %s, %s)",
            (employee_id, email, hashed_password, role)
        )
        # Update applicant status to 'Hired'
        cursor.execute("UPDATE applicants SET status = %s WHERE reg_id = %s", ('Hired', reg_id))
        connection.commit()
        # Predict attrition for the new employee
        predict_attrition_for_employee(employee_id)
        cursor.close()
        connection.close()
        send_hire_email(applicant['email'], applicant['name'], email, raw_password)
        return redirect(url_for('applicant_tracker'))

    elif action == 'reject':
        # Update applicant status to 'Rejected'
        cursor.execute("UPDATE applicants SET status = %s WHERE reg_id = %s", ('Rejected', reg_id))
        connection.commit()
        cursor.close()
        connection.close()
        flash('Applicant rejected.', 'info')
        return redirect(url_for('applicant_tracker'))

    cursor.close()
    connection.close()
    flash('Invalid action.', 'error')
    return redirect(url_for('applicant_tracker'))

@app.route('/applicant-tracker')
@login_required
@role_required('hr')
def applicant_tracker():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT reg_id, name, email, position, department, status
        FROM applicants
        ORDER BY reg_id DESC
    """)
    applicants = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template("hr/applicant-tracker.html", applicants=applicants, employees=[])

@app.route('/employee_view')
def employee_view():
    employee = get_employee_by_id(session.get('employee_id'))
    # Prompt if profile details are missing
    required_fields = ['dob', 'edufield', 'total_working_years', 'distance', 'marital_status', 'gender']
    if any(not employee.get(field) for field in required_fields):
        flash('Please complete your profile details first!', 'warning')
    age = calculate_age(employee.get('dob'))
    return render_template("emp/employee_view.html",
        employee_id=employee['employee_id'],
        name=employee['name'],
        dob=employee.get('dob', ''),
        age=age,
        position=employee.get('position', ''),
        department=employee.get('department', ''),
        edufield=employee.get('edufield', ''),
        salary=employee.get('salary', ''),
        distance=employee.get('distance', ''),
        years_at_company=employee.get('years_at_company', ''),
        marital_status=employee.get('marital_status', ''),
        gender=employee.get('gender', '')
    )

@app.route('/emp_profile')
def emp_profile():
    return render_template("emp/emp_profile.html",
                            name=session.get('name'),
                            position=session.get('position')
                            )

@app.route('/emp_edit_details', methods=['GET', 'POST'], endpoint='emp_edit_details')
@login_required
def emp_edit_details():
    employee_id = session.get('employee_id')
    if not employee_id:
        abort(403)
    if request.method == 'POST':
        dob = request.form.get('dob')
        education_field = request.form.get('education_field')
        total_working_years = request.form.get('total_working_years')
        distance_from_home = request.form.get('distance_from_home')
        marital_status = request.form.get('marital_status')
        gender = request.form.get('gender')

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE employees
            SET dob = %s,
                edufield = %s,
                total_working_years = %s,
                distance = %s,
                marital_status = %s,
                gender = %s
            WHERE employee_id = %s
        """, (dob, education_field, total_working_years, distance_from_home, marital_status, gender, employee_id))
        connection.commit()
        cursor.close()
        connection.close()
        # Predict attrition for this employee after update
        predict_attrition_for_employee(employee_id)
        flash_attrition_status(employee_id)
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('employee_view'))

    employee = get_employee_by_id(employee_id)
    return render_template('emp/emp_edit_details.html', employee=employee)

@app.route('/feedback_portal', methods=['GET', 'POST'])
@login_required
def feedback_portal():
    employee_id = session.get('employee_id')
    employee = get_employee_by_id(employee_id)
    # Check if employee has worked for over 1 year
    years_at_company = employee.get('years_at_company', 0)
    try:
        years_at_company = float(years_at_company)
    except Exception:
        years_at_company = 0

    if request.method == 'POST':
        if years_at_company < 1:
            flash('You need to be in the company for 1 year to submit feedback.', 'error')
            return redirect(url_for('employee_view'))
        job_satisfaction = request.form.get('job_satisfaction')
        overtime = request.form.get('overtime')
        env_satisfaction = request.form.get('env_satisfaction')
        job_involvement = request.form.get('job_involvement')
        job_level = request.form.get('job_level')
        work_life_balance = request.form.get('work_life_balance')

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE employees
            SET job_satisfaction = %s,
                overtime = %s,
                env_satisfaction = %s,
                job_involvement = %s,
                job_level = %s,
                work_life_balance = %s
            WHERE employee_id = %s
        """, (
            job_satisfaction, overtime, env_satisfaction,
            job_involvement, job_level, work_life_balance, employee_id
        ))
        connection.commit()
        cursor.close()
        connection.close()
        # Predict attrition for this employee after feedback update
        predict_attrition_for_employee(employee_id)
        flash_attrition_status(employee_id)
        flash('Feedback submitted successfully.', 'success')
        return redirect(url_for('employee_view'))
    return render_template('emp/feedback_portal.html')

@app.route('/emp_leave', methods=['GET', 'POST'])
def emp_leave():
    if request.method == 'POST':
        leave_type = request.form['leave_type']
        duration = request.form['duration']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        reason = request.form['reason']

        # Validate dates
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            # # Check if dates are in the future
            # if start_date_obj < datetime.now() - timedelta(days=1):
            #     flash('Leave start date cannot be in the past.', 'error')
            #     return redirect(url_for('emp_leave'))
            
            # Check if end date is after start date
            if end_date_obj < start_date_obj:
                flash('End date cannot be before start date.', 'error')
                return redirect(url_for('emp_leave'))

            # Calculate requested days
            requested_days = calculate_leave_days(start_date, end_date, duration)
            
            # Check remaining leave days
            remaining_days = get_remaining_leave_days(session.get('employee_id'), leave_type)
            if requested_days > remaining_days:
                flash(f'You only have {remaining_days} days of {leave_type} remaining.', 'error')
                return redirect(url_for('emp_leave'))

            # Insert leave request into the database
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO leave_requests (employee_id, type, duration, start_date, end_date, reason, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (session.get('employee_id'), leave_type, duration, start_date, end_date, reason, 'Pending')
            )
            connection.commit()
            cursor.close()
            connection.close()
            flash('Leave request submitted successfully.', 'success')
            return redirect(url_for('emp_leave'))
        except Exception as e:
            flash('Invalid leave request. Please check the dates and try again.', 'error')
            return redirect(url_for('emp_leave'))

    # GET request - fetch existing leave requests and remaining days
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Get leave requests
    cursor.execute("""
        SELECT * FROM leave_requests 
        WHERE employee_id = %s 
        ORDER BY start_date DESC
    """, (session.get('employee_id'),))
    leave_requests = cursor.fetchall()
    
    # Calculate total days for each request
    for leave_request in leave_requests:
        leave_request['total_days'] = calculate_leave_days(
            leave_request['start_date'], 
            leave_request['end_date'], 
            leave_request['duration']
        )
    
    # Get remaining days for each leave type
    remaining_days = {
        leave_type: get_remaining_leave_days(session.get('employee_id'), leave_type)
        for leave_type in LEAVE_LIMITS.keys()
    }
    
    connection.close()

    return render_template(
        "emp/emp_leave.html",
        name=session.get('name'),
        position=session.get('position'),
        leave_requests=leave_requests,
        remaining_days=remaining_days,
        leave_limits=LEAVE_LIMITS
    )

@app.route('/update-leave-status/<int:request_id>/<status>')
def update_leave_status(request_id, status):
    if 'user_id' not in session or session.get('role') != 'hr':
        return redirect(url_for('login'))
    
    if status not in ['Approved', 'Rejected', 'Pending']:
        return redirect(url_for('leave_requests'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            UPDATE leave_requests 
            SET status = %s 
            WHERE leave_id = %s
        """, (status, request_id))
        connection.commit()

        # Fetch employee email and leave details
        cursor.execute("""
            SELECT lr.start_date, lr.end_date, e.name
            FROM leave_requests lr
            JOIN employees e ON lr.employee_id = e.employee_id
            WHERE lr.leave_id = %s
        """, (request_id,))
        leave_details = cursor.fetchone()

        if status == 'Approved':
            send_approval_email('021bim061@sxc.edu.np', leave_details)
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('leave_requests'))

def send_approval_email(employee_email, leave_details):
    msg = Message('Leave Request Approved',
                  recipients=[employee_email])
    msg.body = f"Dear {leave_details['name']},\n\nYour leave request for {leave_details['start_date']} to {leave_details['end_date']} has been approved.\n\nBest regards,\nHR Team"
    mail.send(msg)

@app.route('/predict-attrition', methods=['GET', 'POST'])
@login_required
@role_required('hr')
def predict_attrition():
    # Predict and update attrition for all employees
    predict_attrition_for_all()
    # Fetch employees with attrition_risk = 1
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE attrition_risk = 1")
    high_risk_employees = cursor.fetchall()

    # Aggregate factors
    factor_totals = defaultdict(float)
    for emp in high_risk_employees:
        factors = emp.get('attrition_factors')
        if factors:
            try:
                factor_list = json.loads(factors)
                for factor, importance in factor_list:
                    factor_totals[factor] += importance
            except (json.JSONDecodeError, TypeError):
                continue

    # Sort and normalize
    sorted_factors = sorted(factor_totals.items(), key=lambda x: x[1], reverse=True)
    total_importance = sum(val for _, val in sorted_factors) or 1
    factors_chart_data = {
        "labels": [str(f) for f, _ in sorted_factors],
        "data": [round((v / total_importance) * 100, 2) for _, v in sorted_factors]
    }

    # Attrition by Gender
    cursor.execute("""
        SELECT gender, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY gender
    """)
    attrition_gender = cursor.fetchall()
    attrition_gender_labels = [str(row['gender'] or 'Unknown') for row in attrition_gender]
    attrition_gender_data = [int(row['count']) for row in attrition_gender]

    # Attrition by Age Group
    cursor.execute("""
        SELECT 
            CASE 
                WHEN age < 25 THEN '18-24'
                WHEN age < 35 THEN '25-34'
                WHEN age < 45 THEN '35-44'
                WHEN age < 55 THEN '45-54'
                ELSE '55+'
            END as age_group,
            COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY age_group
        ORDER BY MIN(age)
    """)
    attrition_age = cursor.fetchall()
    attrition_age_labels = [str(row['age_group']) for row in attrition_age]
    attrition_age_data = [int(row['count']) for row in attrition_age]

    # Attrition by Department
    departments = get_departments()
    format_strings = ','.join(['%s'] * len(departments))
    cursor.execute(f"""
        SELECT department, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1 AND department IN ({format_strings})
        GROUP BY department
    """, tuple(departments))
    attrition_dept = cursor.fetchall()
    attrition_dept_labels = [str(row['department']) for row in attrition_dept]
    attrition_dept_data = [int(row['count']) for row in attrition_dept]

    # Attrition by Salary Band (5k bands)
    cursor.execute("""
        SELECT 
            CONCAT(FLOOR(salary/5000)*5, 'k-', FLOOR(salary/5000)*5+5, 'k') as salary_band,
            COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY salary_band
        ORDER BY MIN(salary)
    """)
    attrition_salary = cursor.fetchall()
    attrition_salary_labels = [str(row['salary_band']) for row in attrition_salary]
    attrition_salary_data = [int(row['count']) for row in attrition_salary]

    # Attrition by Overtime
    cursor.execute("""
        SELECT overtime, COUNT(*) as count
        FROM employees
        WHERE attrition_risk = 1
        GROUP BY overtime
    """)
    attrition_overtime = cursor.fetchall()
    attrition_overtime_labels = [str(row['overtime'] or 'Unknown') for row in attrition_overtime]
    attrition_overtime_data = [int(row['count']) for row in attrition_overtime]

    cursor.close()
    connection.close()

    # Prepare attrition chart data
    attrition_chart_data = {
        'gender': {
            'labels': attrition_gender_labels,
            'data': attrition_gender_data
        },
        'age': {
            'labels': attrition_age_labels,
            'data': attrition_age_data
        },
        'department': {
            'labels': attrition_dept_labels,
            'data': attrition_dept_data
        },
        'salary': {
            'labels': attrition_salary_labels,
            'data': attrition_salary_data
        },
        'overtime': {
            'labels': attrition_overtime_labels,
            'data': attrition_overtime_data
        }
    }

    return render_template(
        "hr/predict-attrition.html",
        employees=high_risk_employees,
        factors_chart_data=factors_chart_data,
        attrition_chart_data=attrition_chart_data
    )

@app.route('/delete-leave-request/<int:leave_id>', methods=['POST'])
@login_required
def delete_leave_request(leave_id):
    # Only allow the employee who owns the leave request to delete it
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM leave_requests WHERE leave_id = %s AND employee_id = %s", (leave_id, session.get('employee_id')))
    leave = cursor.fetchone()
    if not leave:
        flash('You are not authorized to delete this leave request.', 'error')
        cursor.close()
        connection.close()
        return redirect(url_for('emp_leave'))

    # Check if the leave request is approved
    if leave['status'] == 'Approved':
        flash('Cannot delete an approved leave request.', 'error')
        cursor.close()
        connection.close()
        return redirect(url_for('emp_leave'))

    cursor.execute("DELETE FROM leave_requests WHERE leave_id = %s", (leave_id,))
    connection.commit()
    cursor.close()
    connection.close()
    flash('Leave request deleted successfully.', 'success')
    return redirect(url_for('emp_leave'))

def get_anniversaries_this_month(employees):
    """Return a list of employees whose work anniversaries are in the current month, with labels for today/tomorrow."""
    today = datetime.today()
    current_month = today.month
    anniversaries = []
    for emp in employees:
        date_joined = emp.get('date_joined')
        if not date_joined or date_joined in ('', None):
            continue
        # Defensive: handle both string and datetime
        if isinstance(date_joined, str):
            try:
                joined_date = datetime.strptime(date_joined, '%Y-%m-%d')
            except Exception:
                continue
        else:
            joined_date = date_joined
        if joined_date.month == current_month:
            anniv_day = joined_date.day
            # Calculate years at company
            years = today.year - joined_date.year - ((today.month, today.day) < (joined_date.month, joined_date.day))
            # Make a copy so we don't mutate the original
            emp_copy = dict(emp)
            emp_copy['years_at_company'] = years
            # Annotate if today or tomorrow
            if anniv_day == today.day:
                emp_copy['anniversary_label'] = 'Today'
            elif anniv_day == (today + timedelta(days=1)).day:
                emp_copy['anniversary_label'] = 'Tomorrow'
            else:
                emp_copy['anniversary_label'] = ''
            anniversaries.append(emp_copy)
    # Sort by anniversary day, show up to 5
    anniversaries = sorted(
        anniversaries,
        key=lambda e: datetime.strptime(str(e.get('date_joined')), '%Y-%m-%d').day if e.get('date_joined') else 0
    )[:5]
    return anniversaries

def get_birthdays_this_month(employees):
    """Return a list of employees whose birthdays are in the current month, with labels for today/tomorrow."""
    today = datetime.today()
    current_month = today.month
    birthdays = []
    for emp in employees:
        dob = emp.get('dob')
        if not dob or dob in ('', None):
            continue
        # Defensive: handle both string and datetime
        if isinstance(dob, str):
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d')
            except Exception:
                continue
        else:
            dob_date = dob
        if dob_date.month == current_month:
            bday_day = dob_date.day
            # Calculate age
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
            emp_copy = dict(emp)
            emp_copy['age'] = age
            # Annotate if today or tomorrow
            if bday_day == today.day:
                emp_copy['birthday_label'] = 'Today'
            elif bday_day == (today + timedelta(days=1)).day:
                emp_copy['birthday_label'] = 'Tomorrow'
            else:
                emp_copy['birthday_label'] = ''
            birthdays.append(emp_copy)
    # Sort by birthday day, show up to 5
    birthdays = sorted(
        birthdays,
        key=lambda e: datetime.strptime(str(e.get('dob')), '%Y-%m-%d').day if e.get('dob') else 0
    )[:5]
    return birthdays

@app.route('/edit_applicant/<int:reg_id>', methods=['POST'])
@login_required
@role_required('hr')
def edit_applicant(reg_id):
    name = request.form.get('name')
    email = request.form.get('email')
    position = request.form.get('position')
    department = request.form.get('department')
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE applicants
        SET name = %s, email = %s, position = %s, department = %s
        WHERE reg_id = %s
    """, (name, email, position, department, reg_id))
    connection.commit()
    cursor.close()
    connection.close()
    flash('Applicant updated successfully.', 'success')
    return redirect(url_for('applicant_tracker'))

@app.route('/delete_applicant/<int:reg_id>', methods=['POST'])
@login_required
@role_required('hr')
def delete_applicant(reg_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM applicants WHERE reg_id = %s", (reg_id,))
    connection.commit()
    cursor.close()
    connection.close()
    flash('Applicant deleted successfully.', 'success')
    return redirect(url_for('applicant_tracker'))

@app.template_filter('to_json')
def to_json(value):
    if value is None:
        return '[]'
    try:
        return json.dumps(value)
    except:
        return '[]'

@app.template_filter('from_json')
def from_json(value):
    if value is None:
        return []
    try:
        return json.loads(value)
    except:
        return []

def ensure_attrition_factors_column():
    """Ensure the attrition_factors and attrition_probability columns exist in the employees table."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        # Check if attrition_factors column exists
        cursor.execute("SHOW COLUMNS FROM employees LIKE 'attrition_factors'")
        if not cursor.fetchone():
            # Add the column if it doesn't exist
            cursor.execute("ALTER TABLE employees ADD COLUMN attrition_factors JSON")
            connection.commit()
        
        # Check if attrition_probability column exists
        cursor.execute("SHOW COLUMNS FROM employees LIKE 'attrition_probability'")
        if not cursor.fetchone():
            # Add the column if it doesn't exist
            cursor.execute("ALTER TABLE employees ADD COLUMN attrition_probability DECIMAL(5,4)")
            connection.commit()
    finally:
        cursor.close()
        connection.close()

def update_employee_shap_explanations(employee_id):
    """Update SHAP explanations for an employee after their data changes."""
    try:
        predict_attrition_for_employee(employee_id)
        return True
    except Exception as e:
        print(f"Error updating SHAP explanations for employee {employee_id}: {e}")
        return False

def flash_attrition_status(employee_id):
    """Flash a message about the employee's attrition status after prediction."""
    emp = get_employee_by_id(employee_id)
    if emp and emp.get('attrition_risk') == 1:
        flash(f'Warning: {emp["name"]} is at high risk of attrition.', 'warning')
    elif emp and emp.get('attrition_risk') == 0:
        flash(f'{emp["name"]} is at low risk of attrition.', 'success')

# Call this when the app starts
ensure_attrition_factors_column()

@app.route('/get_explanation/<int:employee_id>')
@login_required
@role_required('hr')
def get_explanation(employee_id):
    """Get SHAP explanation data for an employee."""
    employee = get_employee_by_id(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
        
    try:
        factors = json.loads(employee.get('attrition_factors', '[]'))
        return jsonify({
            'name': employee['name'],
            'probability': employee.get('attrition_probability', 0),
            'factors': factors
        })
    except Exception as e:
        print(f"Error getting explanation: {e}")
        return jsonify({'error': 'Failed to get explanation'}), 500

if __name__ == '__main__':
    app.run(debug=True)