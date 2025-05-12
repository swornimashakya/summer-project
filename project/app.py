from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
import random # For simulating data, remove in production
import mysql.connector
from datetime import datetime
import bcrypt  # For password hashing
from functools import wraps
from flask_mail import Mail, Message

app = Flask(__name__)

app.secret_key = 's3cr3tk3y'  # Secret key for session management

# Configure email settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your mail server
app.config['MAIL_PORT'] = 587  # Common port for TLS
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '021bim061@sxc.edu.np'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'sxc0sxc@#'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'HR Manager <021bim061@sxc.edu.np>'  # Replace with your email

mail = Mail(app)

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
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="employee_db",
        auth_plugin='mysql_native_password'
    )
    return connection

def get_leave_requests():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT lr.*, e.name, e.department as dept 
        FROM leave_requests lr
        JOIN employees e ON lr.emp_id = e.id
        ORDER BY lr.start_date DESC
    """)
    leave_requests = cursor.fetchall()
    
    # Calculate total days for each leave request
    for request in leave_requests:
        # Convert date strings to datetime objects
        start_date_obj = datetime.strptime(str(request['start_date']), '%Y-%m-%d')
        end_date_obj = datetime.strptime(str(request['end_date']), '%Y-%m-%d')
        
        # Calculate total days
        total_days = (end_date_obj - start_date_obj).days + 1
        
        # For half day leave
        if request['duration'] == 'Half Day':
            total_days = total_days / 2
            
        # Add total_days to the request dictionary
        request['total_days'] = total_days

    return leave_requests

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

        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        session['user_id'] = user['id']
        session['role'] = user['role']
        session['employee_id'] = user['employee_id']
        session['email'] = user['email']
        
        # Fetch employee details
        cursor.execute("SELECT name, position FROM employees WHERE id = %s", (user['employee_id'],))
        employee = cursor.fetchone()
        session['name'] = employee['name']
        session['position'] = employee['position']
        
        if user['role'] == 'hr':
            return redirect(url_for('index'))
        elif user['role'] == 'employee':
            return redirect(url_for('employee_view'))
        else:
            flash('Invalid user role', 'error')
            return redirect(url_for('login'))

        cursor.close()
        connection.close()
    else:
        return render_template('login.html')

@app.route('/index')
@login_required
@role_required('hr')
def index():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch employee data from the database
    cursor.execute("SELECT COUNT(*) AS total FROM employees")
    total_employees = cursor.fetchone()['total']
    
    # Simulate a random attrition rate (replace with actual model prediction logic)
    attrition_rate = random.uniform(10, 30)  # Just a random number for now

    # Fetch leave requests data from the database
    leave_requests = get_leave_requests()

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
                
    return render_template('index.html',
                        attrition_rate=attrition_rate,
                        total_employees=total_employees,
                        name=session.get('name'),
                        position=session.get('position'),
                        leave_requests=leave_requests,
                        employees=employees,
                        department_labels=department_labels,
                        department_data=department_data,
                        age_labels=age_labels,
                        age_data=age_data
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
    # Connect to the database
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch employee data from the database
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    
    # Close the database connection
    cursor.close()
    connection.close()
    return render_template("employees.html", employees=employees)

@app.route('/employees/add_employee', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        position = request.form['position']
        department = request.form['department']
        salary = request.form['salary']
        status = request.form['status']
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
            "INSERT INTO employees (name, position, department, salary, status, years_at_company) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (name, position, department, salary, status, years_at_company)
        )
        
        connection.commit()
        cursor.close()
        connection.close()

        flash('Employee added successfully', 'success')
        return redirect(url_for('employees'))
    return render_template('add_employee.html')

@app.route('/employees/edit_employee/<int:employee_id>', methods=['GET', 'POST'])
def edit_employee(employee_id):
    if request.method == 'POST':
        name = request.form['name']
        position = request.form['position']
        department = request.form['department']
        salary = request.form['salary']
        status = request.form['status']
        years_at_company = request.form['years_at_company']
        # age = request.form['age']
        # marital_status = request.form['marital_status']

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE employees
            SET name = %s, position = %s, department = %s, salary = %s, status = %s, years_at_company = %s 
            WHERE id = %s 
        """, (name, position, department, salary, status, years_at_company, employee_id))

        connection.commit()
        cursor.close()
        connection.close()

        flash('Employee updated successfully', 'success')
        return redirect(url_for('employees'))
    
    # Fetch employee data for GET request
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE id = %s", (employee_id,))
    employee = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))
        
    return render_template('edit_employee.html', employee=employee)
        
@app.route('/employees/delete_employee/<int:employee_id>') 
def delete_employee(employee_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
    connection.commit()
    cursor.close()
    connection.close()

    flash('Employee deleted successfully', 'success')
    return redirect(url_for('employees'))

@app.route('/department-data')
def department_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Department distribution
    cursor.execute("SELECT department, COUNT(*) as count FROM employees GROUP BY department")
    dept_data = cursor.fetchall()
    
    conn.close()
    return jsonify({
        'labels': [item['department'] for item in dept_data],
        'counts': [item['count'] for item in dept_data]
    })

@app.route('/age-data')
def age_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Age ranges (20-30, 31-40, etc.)
    cursor.execute("""
        SELECT 
            CASE
                WHEN age BETWEEN 18 AND 20 THEN '18-20' 
                WHEN age BETWEEN 20 AND 30 THEN '20-30'
                WHEN age BETWEEN 31 AND 40 THEN '31-40'
                WHEN age BETWEEN 41 AND 50 THEN '41-50'
                ELSE '50+'
            END as age_range,
            COUNT(*) as count
        FROM employees
        GROUP BY age_range
        ORDER BY age_range
    """)
    age_data = cursor.fetchall()
    
    conn.close()
    return jsonify({
        'labels': [item['age_range'] for item in age_data],
        'counts': [item['count'] for item in age_data]
    })

@app.route('/salary-data')
def salary_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Salary bands (in $10k increments)
    cursor.execute("""
    SELECT 
        CONCAT(FLOOR(salary/10000)*10, 'k-', FLOOR(salary/10000)*10+10, 'k') as salary_band,
        COUNT(*) as count
    FROM employees
    GROUP BY salary_band
    ORDER BY MIN(salary)
    """)
    salary_data = cursor.fetchall()

    return jsonify({
        'labels': [row['salary_band'] for row in salary_data],
        'counts': [row['count'] for row in salary_data]
})

@app.route('/leave-requests')
def leave_requests():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    leave_requests = get_leave_requests()
    return render_template("leave_requests.html",
                           leave_requests=leave_requests
                           )

@app.route('/reports')
@login_required
@role_required('hr')
def reports():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Get employee counts by status
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as active,
            SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as on_leave,
            SUM(CASE WHEN status = 'Left' THEN 1 ELSE 0 END) as `left`
        FROM employees
    """)
    counts = cursor.fetchone()
    
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
    
    return render_template('reports.html',
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
    # Fetch employee data from the database
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE id = %s", (session.get('employee_id'),))
    employee = cursor.fetchone()
    connection.close()
    return render_template("employee_view.html",
                            employee_id=session.get('employee_id'),
                            name=session.get('name'),
                            position=session.get('position'),
                            department=employee['department'],
                            salary=employee['salary'],
                            status=employee['status'],
                            years_at_company=employee['years_at_company'],
                            age=employee['age'],
                            marital_status=employee['marital_status'],
                            email=session.get('email')
                            )

@app.route('/emp_profile')
def emp_profile():
    return render_template("emp_profile.html",
                            name=session.get('name'),
                            position=session.get('position')
                            )

@app.route('/emp_leave', methods=['GET', 'POST'])
def emp_leave():
    if request.method == 'POST':
        leave_type = request.form['leave_type']
        duration = request.form['duration']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        reason = request.form['reason']
        
        # Insert leave request into the database
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Convert date strings to datetime objects
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate total days
        total_days = (end_date_obj - start_date_obj).days + 1
        
        # For half day leave
        if duration == 'Half Day':
            total_days = total_days / 2
        
        # Insert leave request into the database
        cursor.execute("INSERT INTO leave_requests (emp_id, type, duration, start_date, end_date, reason, status) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                       (session.get('employee_id'), leave_type, duration, start_date_obj, end_date_obj, reason, 'Pending'))
        
        connection.commit()
        cursor.close()
        connection.close()  
        
        return redirect(url_for('emp_leave'))

    # GET request - fetch existing leave requests
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch leave requests for the current employee
    cursor.execute("SELECT * FROM leave_requests WHERE emp_id = %s ORDER BY `start_date` DESC", 
                  (session.get('employee_id'),))
    leave_requests = cursor.fetchall()
    
    # Calculate total days for each leave request
    for leave_request in leave_requests:
        start_date = leave_request['start_date']
        end_date = leave_request['end_date']
        duration = leave_request['duration']
        
        # Calculate total days
        total_days = (end_date - start_date).days + 1
        
        # For half day leave
        if duration == 'Half Day':
            total_days = total_days / 2
            
        # Add total_days to the request dictionary
        leave_request['total_days'] = total_days
    
    connection.close()

    return render_template("emp_leave.html",
                         name=session.get('name'),
                         position=session.get('position'),
                         leave_requests=leave_requests)

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
            SELECT lr.start_date, lr.end_date
            FROM leave_requests lr
            JOIN employees e ON lr.emp_id = e.id
            WHERE lr.leave_id = %s
        """, (request_id,))
        leave_details = cursor.fetchone()

        if status == 'Approved':
            send_approval_email('swornima9@gmail.com', leave_details)
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('leave_requests'))

@app.route('/attrition-prediction')
def attrition_prediction():
    return render_template("attrition_prediction.html")

def send_approval_email(employee_email, leave_details):
    msg = Message('Leave Request Approved',
                  recipients=[employee_email])
    msg.body = f"Dear Employee,\n\nYour leave request for {leave_details['start_date']} to {leave_details['end_date']} has been approved.\n\nBest regards,\nHR Team"
    mail.send(msg)

if __name__ == "__main__":
    app.run(debug=True)