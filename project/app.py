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
            cursor.execute("SELECT name, position FROM employees WHERE id = %s", (user['employee_id'],))
            employee = cursor.fetchone()
            cursor.close()
            connection.close()

            if employee:
                session['name'] = employee['name']
                session['position'] = employee['position']
                session['user_id'] = user['id']
                return redirect(url_for('index'))
        else:
            cursor.close()
            connection.close()
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@app.route('/index')
def index():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    # Fetch employee data from the database
    cursor.execute("SELECT COUNT(*) AS total FROM employees")
    total_employees = cursor.fetchone()['total']
    
    # Simulate a random attrition rate (replace with actual model prediction logic)
    attrition_rate = random.uniform(10, 30)  # Just a random number for now
    return render_template('index.html',
                           attrition_rate=attrition_rate,
                           total_employees=total_employees
                           )

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

@app.route('/reports')
def reports():

    return render_template("reports.html")

# # API endpoint for employee distribution data
# @app.route('/get_data')
# def get_data():
#     # Simulate data (replace with your actual model predictions or data)
#     data = {
#         "labels": ["Sales", "R&D", "HR"],  # Department names
#         "values": [50, 70, 30]  # Number of employees in each department
#     }
#     return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
