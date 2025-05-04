import mysql.connector

print("Connecting to MySQL database...")

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345",
    database="employee_db",
    auth_plugin='mysql_native_password'
)

print("Connected successfully!")
connection.close()
