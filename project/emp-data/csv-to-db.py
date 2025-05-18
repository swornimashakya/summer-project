import csv
import mysql.connector
from mysql.connector import Error

try:
    # Connect to the database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="employee_db"
    )

    if connection.is_connected():
        cursor = connection.cursor()

        # Insert employee data into employees table
        with open(r'D:\BIM\Summer Project\project\employees.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Print the row to verify column names
                print(row)

                # Insert into employees table and get employee_id (auto-increment)
                cursor.execute("""
                    INSERT INTO employees (name, age, position, department, salary, total_working_years, marital_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (row['name'], int(row['age']), row['position'], row['department'], float(row['salary']), int(row['total_working_years']), row['marital_status']))

                # Get the last inserted employee_id
                employee_id = cursor.lastrowid

                # Insert into users table with the correct employee_id
                cursor.execute("""
                    INSERT INTO users (email, password, role, employee_id)
                    VALUES (%s, %s, %s, %s)
                """, (row['email'], row['password'], row['role'], employee_id))

        # Commit the transaction and close the connection
        connection.commit()

except Error as e:
    print(f"Error: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed.")

