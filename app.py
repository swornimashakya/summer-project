from flask import Flask, render_template, jsonify
import random # For simulating data, remove in production
import mysql.connector

app = Flask(__name__)

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

@app.route('/')
def index():

    # Simulate a random attrition rate (replace with actual model prediction logic)
    attrition_rate = random.uniform(10, 30)  # Just a random number for now
    return render_template('index.html', attrition_rate=attrition_rate)

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
    return render_template("employees.html", employees=employees)

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
