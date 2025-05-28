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
    
    return int(total_days)

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
    cursor.execute("SELECT COUNT(*) AS at_risk_count FROM employees WHERE attrition_risk = 1")
    at_risk_count = cursor.fetchone()['at_risk_count']

    # Fetch employee data from the database
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()

    # Get department distribution
    cursor.execute("""
        SELECT department, COUNT(*) as count
        FROM employees
        GROUP BY department
    """)
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
    age_data = [row['count'] for row in age_data]
                
    return render_template('hr/dashboard.html',
                        attrition_rate=attrition_rate,
                        total_employees=total_employees,
                        name=session.get('name'),
                        position=session.get('position'),
                        leave_requests=leave_requests,
                        employees=employees,
                        department_labels=department_labels,
                        department_data=department_data,
                        age_labels=age_labels,
                        age_data=age_data,
                        at_risk_count=at_risk_count
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

        cursor.execute(
            "INSERT INTO employees (name, position, department, salary, years_at_company) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (name, position, department, salary, years_at_company)
        )
        
        connection.commit()
        cursor.close()
        connection.close()

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
        years_at_company = request.form['years_at_company']
        # age = request.form['age']
        # marital_status = request.form['marital_status']

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE employees
            SET name = %s, position = %s, department = %s, salary = %s
            , years_at_company = %s 
            WHERE employee_id = %s 
        """, (name, position, department, salary, years_at_company, employee_id))

        connection.commit()
        cursor.close()
        connection.close()

        flash('Employee updated successfully', 'success')
        return redirect(url_for('employees'))
    
    employee = get_employee_by_id(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))
        
    return render_template('hr/edit_employee.html', employee=employee)
        
@app.route('/employees/delete_employee/<int:employee_id>') 
def delete_employee(employee_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM employees WHERE employee_id = %s", (employee_id,))
    connection.commit()
    cursor.close()
    connection.close()

    flash('Employee deleted successfully', 'success')
    return redirect(url_for('employees'))

def get_chart_data(query_type):
    """Fetch chart data based on the query type."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    if query_type == 'department':
        cursor.execute("SELECT department, COUNT(*) as count FROM employees GROUP BY department")
        data = cursor.fetchall()
        labels = [item['department'] for item in data]
        counts = [item['count'] for item in data]
    
    elif query_type == 'age':
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
        data = cursor.fetchall()
        labels = [item['age_group'] for item in data]
        counts = [item['count'] for item in data]
    
    elif query_type == 'salary':
        cursor.execute("""
            SELECT 
                CONCAT(FLOOR(salary/10000)*10, 'k-', FLOOR(salary/10000)*10+10, 'k') as salary_band,
                COUNT(*) as count
            FROM employees
            GROUP BY salary_band
            ORDER BY MIN(salary)
        """)
        data = cursor.fetchall()
        labels = [item['salary_band'] for item in data]
        counts = [item['count'] for item in data]
    
    else:
        labels, counts = [], []
    
    cursor.close()
    connection.close()
    return labels, counts

@app.route('/department-data')
def department_data():
    labels, counts = get_chart_data('department')
    return jsonify({'labels': labels, 'counts': counts})

@app.route('/age-data')
def age_data():
    labels, counts = get_chart_data('age')
    return jsonify({'labels': labels, 'counts': counts})

@app.route('/salary-data')
def salary_data():
    labels, counts = get_chart_data('salary')
    return jsonify({'labels': labels, 'counts': counts})

@app.route('/leave-requests')
def leave_requests():
    leave_requests = get_leave_requests()
    return render_template("hr/leave_requests.html",
                           leave_requests=leave_requests,
                           employees=[]
                           )

@app.route('/reports')
@login_required
@role_required('hr')
def reports():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Get department distribution
    cursor.execute("""
        SELECT department, COUNT(*) as count
        FROM employees
        GROUP BY department
    """)
    dept_data = cursor.fetchall()
    department_labels = [row['department'] for row in dept_data]
    department_data = [row['count'] for row in dept_data]
    
    # Get leave request trends (last 6 months)
    cursor.execute("""
        SELECT DATE_FORMAT(start_date, '%Y-%m') as month, COUNT(*) as count
        FROM leave_requests
        WHERE start_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        GROUP BY DATE_FORMAT(start_date, '%Y-%m')
        ORDER BY month
    """)
    leave_trends = cursor.fetchall()
    leave_trends_labels = [row['month'] for row in leave_trends]
    leave_trends_data = [row['count'] for row in leave_trends]
    
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
    age_data = [row['count'] for row in age_data]
    
    # Get leave type distribution
    cursor.execute("""
        SELECT type, COUNT(*) as count
        FROM leave_requests
        GROUP BY type
    """)
    leave_type_data = cursor.fetchall()
    leave_type_labels = [row['type'] for row in leave_type_data]
    leave_type_data = [row['count'] for row in leave_type_data]
    
    cursor.close()
    connection.close()
    
    return render_template('hr/reports.html',
                         total_employees=counts['total'],
                         active_employees=counts['active'],
                         on_leave=counts['on_leave'],
                         left=counts['left'],
                         department_labels=department_labels,
                         department_data=department_data,
                         leave_trends_labels=leave_trends_labels,
                         leave_trends_data=leave_trends_data,
                         age_labels=age_labels,
                         age_data=age_data,
                         leave_type_labels=leave_type_labels,
                         leave_type_data=leave_type_data)

@app.route('/employee_view')
def employee_view():
    employee = get_employee_by_id(session.get('employee_id'))
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
        # Predict attrition for all employees after update
        predict_attrition_for_all()
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
        # Predict attrition for all employees after feedback update
        predict_attrition_for_all()
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

def preprocess_employee_data(raw_data):
    """
    raw_data: dict or DataFrame with the same columns as in the database
    Returns: DataFrame ready for prediction
    """
    # Standardize keys to match database (lowercase)
    data = {k.lower(): v for k, v in raw_data.items()}
    df = pd.DataFrame([data])

    # Binary encoding with safe get
    df['gender'] = df.get('gender', pd.Series([None]))
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    df['overtime'] = df.get('overtime', pd.Series([None]))
    df['overtime'] = df['overtime'].map({'Yes': 1, 'No': 0})

    df['attrition'] = df.get('attrition', 0)  # If present

    # One-hot encoding (must match training)
    for col, prefix in [('department', 'Dept'), ('educationfield', 'EduField'), ('jobrole', 'JobRole')]:
        if col in df:
            df = df.join(pd.get_dummies(df[col], prefix=prefix)).drop(col, axis=1)
    if 'maritalstatus' in df:
        df = df.join(pd.get_dummies(df['maritalstatus'])).drop('maritalstatus', axis=1)

    # Ensure all columns exist and are in the correct order
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns as zeros

    df = df[model_columns]  # Reorder columns

    # --- Apply the scaler loaded from encoders.pkl ---
    scaler = encoder_data.get('scaler')
    if scaler:
        df = pd.DataFrame(scaler.transform(df), columns=model_columns)

    return df

def predict_attrition_for_all():
    """Predict attrition for all employees and update the database."""
    employees = get_all_employees()
    connection = get_db_connection()
    cursor = connection.cursor()
    for emp in employees:
        X = preprocess_employee_data(emp)
        prediction = int(model.predict(X)[0])
        # If you want probability, uncomment below:
        # probability = float(model.predict_proba(X)[0][1])
        cursor.execute("""
            UPDATE employees
            SET attrition_risk = %s
            WHERE employee_id = %s
        """, (prediction, emp['employee_id']))
    connection.commit()
    cursor.close()
    connection.close()

@app.route('/predict-attrition')
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

if __name__ == '__main__':
    app.run(debug=True)