from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash, abort
import random # For simulating data, remove in production
import mysql.connector
from datetime import datetime, timedelta
import bcrypt  # For password hashing
from functools import wraps
from flask_mail import Mail, Message
import config  # Import the config module
import pickle
import pandas as pd
from calendar import monthrange

app = Flask(__name__)

app.secret_key = config.SECRET_KEY # Use the secret key from config

app.config.update(config.MAIL_CONFIG) # Configure email settings using config

mail = Mail(app)

MODEL_PATH = 'D:/BIM/Summer Project/project/models/random-forest-model.pkl'
ENCODER_PATH = 'D:/BIM/Summer Project/project/models/encoders.pkl'

with open(MODEL_PATH, 'rb') as f:
    model, model_columns = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    encoder_data = pickle.load(f)
    model_columns = encoder_data['columns']

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

def predict_attrition_for_employee(employee_id):
    """Predict attrition for a single employee and update the database."""
    emp = get_employee_by_id(employee_id)
    if not emp:
        return
    X = preprocess_employee_data(emp)
    prediction = int(model.predict(X)[0])
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE employees
        SET attrition_risk = %s
        WHERE employee_id = %s
    """, (prediction, employee_id))
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
        cursor.execute("""
            UPDATE employees
            SET attrition_risk = %s
            WHERE employee_id = %s
        """, (prediction, emp['employee_id']))
    connection.commit()
    cursor.close()
    connection.close()

@app.route('/', methods=['GET', 'POST'])
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
    attrition_age_labels = [row['age_group'] for row in attrition_age]
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
        # age = request.form['age']
        # marital_status = request.form['marital_status']

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
        flash_attrition_status(employee_id)
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

# @app.route('/reports')
# @login_required
# @role_required('hr')
# def reports():
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     # Get department distribution dynamically
#     departments = get_departments()
#     format_strings = ','.join(['%s'] * len(departments))
#     cursor.execute(f"""
#         SELECT department, COUNT(*) as count
#         FROM employees
#         WHERE department IN ({format_strings})
#         GROUP BY department
#     """, tuple(departments))
#     dept_data = cursor.fetchall()
#     department_labels = [row['department'] for row in dept_data]
#     department_data = [row['count'] for row in dept_data]
    
#     # Get leave request trends (last 6 months)
#     cursor.execute("""
#         SELECT DATE_FORMAT(start_date, '%Y-%m') as month, COUNT(*) as count
#         FROM leave_requests
#         WHERE start_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
#         GROUP BY DATE_FORMAT(start_date, '%Y-%m')
#         ORDER BY month
#     """)
#     leave_trends = cursor.fetchall()
#     leave_trends_labels = [row['month'] for row in leave_trends]
#     leave_trends_data = [row['count'] for row in leave_trends]
    
#     # Get age distribution
#     cursor.execute("""
#         SELECT 
#             CASE 
#                 WHEN age < 25 THEN '18-24'
#                 WHEN age < 35 THEN '25-34'
#                 WHEN age < 45 THEN '35-44'
#                 WHEN age < 55 THEN '45-54'
#                 ELSE '55+'
#             END as age_group,
#             COUNT(*) as count
#         FROM employees
#         GROUP BY age_group
#         ORDER BY MIN(age)
#     """)
#     age_data = cursor.fetchall()
#     age_labels = [row['age_group'] for row in age_data]
#     age_data = [row['count'] for row in age_data]
    
#     # Get leave type distribution
#     cursor.execute("""
#         SELECT type, COUNT(*) as count
#         FROM leave_requests
#         GROUP BY type
#     """)
#     leave_type_data = cursor.fetchall()
#     leave_type_labels = [row['type'] for row in leave_type_data]
#     leave_type_data = [row['count'] for row in leave_type_data]
    
#     cursor.close()
#     connection.close()
    
#     return render_template('hr/reports.html',
#                          total_employees=counts['total'],
#                          active_employees=counts['active'],
#                          on_leave=counts['on_leave'],
#                          left=counts['left'],
#                          department_labels=department_labels,
#                          department_data=department_data,
#                          leave_trends_labels=leave_trends_labels,
#                          leave_trends_data=leave_trends_data,
#                          age_labels=age_labels,
#                          age_data=age_data,
#                          leave_type_labels=leave_type_labels,
#                          leave_type_data=leave_type_data)

@app.route('/employee_view')
def employee_view():
    employee = get_employee_by_id(session.get('employee_id'))
    # Prompt if profile details are missing
    required_fields = ['dob', 'edufield', 'total_working_years', 'distance', 'marital_status', 'gender']
    if any(not employee.get(field) for field in required_fields):
        flash('Please complete your profile details for accurate attrition prediction.', 'warning')
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
    if request.method == 'POST':
        employee_id = session.get('employee_id')
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
            if (start_date_obj and end_date_obj) > (datetime.now() - timedelta(days=1)):

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
                flash('Leave request submitted.', 'success')
                return redirect(url_for('emp_leave'))
            else:
                flash('Invalid leave request. Please check the dates and try again.', 'error')
                return redirect(url_for('emp_leave'))
        except Exception as e:
            flash('Invalid leave request. Please check the dates and try again.', 'error')
            return redirect(url_for('emp_leave'))

    # GET request - fetch existing leave requests
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM leave_requests WHERE employee_id = %s ORDER BY `start_date` DESC", 
                  (session.get('employee_id'),))
    leave_requests = cursor.fetchall()
    # Calculate total days
    for leave_request in leave_requests:
        leave_request['total_days'] = calculate_leave_days(
            leave_request['start_date'], leave_request['end_date'], leave_request['duration']
        )
    connection.close()

    return render_template(
        "emp/emp_leave.html",
        name=session.get('name'),
        position=session.get('position'),
        leave_requests=leave_requests
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
    cursor.close()
    connection.close()
    return render_template(
        "hr/predict-attrition.html",
        employees=high_risk_employees
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

if __name__ == '__main__':
    app.run(debug=True)