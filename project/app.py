from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import random # For simulating data, remove in production
import mysql.connector
from datetime import datetime
import bcrypt  # For password hashing

app = Flask(__name__)

app.secret_key = 's3cr3tk3y'  # Secret key for session management

# Database connection function
def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="employee_db",
        auth_plugin='mysql_native_password'
    )
    return connection

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

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['user_id'] = user['id']
            session['role'] = user['role']

            cursor.execute("SELECT name, position FROM employees WHERE id = %s", (user['employee_id'],))
            employee = cursor.fetchone()
            session['name'] = employee['name']
            session['position'] = employee['position']
            
            if user['role'] == 'hr':
                return redirect(url_for('index'))  # HR sees index
            elif user['role'] == 'employee':
                return redirect(url_for('employee_view'))  # Employee view
            else:
                return render_template('login.html', error="Employee record not found.")
        cursor.close()
        connection.close()
    else:
        return render_template('login.html')

@app.route('/index')
def index():
    if 'user_id' in session:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Fetch employee data from the database
        cursor.execute("SELECT COUNT(*) AS total FROM employees")
        total_employees = cursor.fetchone()['total']
        
        # Simulate a random attrition rate (replace with actual model prediction logic)
        attrition_rate = random.uniform(10, 30)  # Just a random number for now
        return render_template('index.html',
                            attrition_rate=attrition_rate,
                            total_employees=total_employees,
                                name=session.get('name'),
                                position=session.get('position')
                            )
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    session.pop('name', None)
    session.pop('position', None)
    return redirect(url_for('login'))

@app.route('/employees')
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

        # Redirect to the list of employees after adding
        return redirect(url_for('employees'))
    return render_template('add_employee.html')


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

@app.route('/reports')
def reports():

    return render_template("reports.html")

@app.route('/employee_view')
def employee_view():
    return ("Employee View Page")
    
# API endpoint for employee distribution data
@app.route('/get_data')
def get_data():
    # Simulate data (replace with your actual model predictions or data)
    data = {
        "labels": ["Sales", "R&D", "HR"],  # Department names
        "values": [50, 70, 30]  # Number of employees in each department
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
