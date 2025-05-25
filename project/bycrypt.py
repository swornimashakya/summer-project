import csv
import mysql.connector
from mysql.connector import Error
import random
from datetime import datetime, timedelta
import bcrypt

def get_db_connection():
    return mysql.connector.connect(
        user='root',
        password='12345',
        host='localhost',
        database='employee_db',
        auth_plugin='mysql_native_password'
    )

def generate_hashed_password():
    # Hash the default password 'pass123' with bcrypt
    return bcrypt.hashpw(b'pass123', bcrypt.gensalt())

def insert_employee_data(employee_csv, names_csv):
    try:
        # Load names and prepare hashed password
        with open(names_csv) as f:
            names = [line.strip() for line in f if line.strip()]
        hashed_password = generate_hashed_password()
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Clear tables
        cursor.execute("DELETE FROM employees")
        cursor.execute("DELETE FROM users")
        conn.commit()

        # Process employee data
        with open(employee_csv) as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                
                # Assign names in order, cycling if needed
                name = names[idx % len(names)]

                age = int(row['Age'])
                join_date = (datetime.now() - timedelta(days=int(row['YearsAtCompany'])*365)).date()
                
                # Insert employee (without employee_id)
                cursor.execute("""
                    INSERT INTO employees (
                        name, dob, age, position, department, salary,
                        total_working_years, years_at_company, date_joined,
                        marital_status, gender, edufield, env_satisfaction,
                        job_involvement, job_level, job_satisfaction, overtime,
                        work_life_balance, distance, attrition_risk
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    name,
                    datetime(datetime.now().year - age, random.randint(1, 12), random.randint(1, 28)).date(),
                    age,
                    row['JobRole'], row['Department'], float(row['MonthlyIncome']),
                    int(row['TotalWorkingYears']), int(row['YearsAtCompany']),
                    join_date,
                    row['MaritalStatus'], row['Gender'], row['EducationField'],
                    int(row['EnvironmentSatisfaction']), int(row['JobInvolvement']),
                    int(row['JobLevel']), int(row['JobSatisfaction']), row['OverTime'],
                    int(row['WorkLifeBalance']), int(row['DistanceFromHome']),
                    1 if row['Attrition'] == 'Yes' else 0
                ))
                
                # Get auto-generated employee_id
                employee_id = cursor.lastrowid
                
                # Insert user with hashed password
                cursor.execute("""
                    INSERT INTO users (
                        employee_id, email, password, role
                    ) VALUES (%s, %s, %s, %s)
                """, (
                    employee_id,
                    f"{name.lower().replace(' ', '.')}@company.com",
                    hashed_password,
                    'employee'
                ))

        conn.commit()
        print(f"Successfully inserted {len(names)} employees with users")

    except Error as e:
        print(f"Database error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def hash_all_user_passwords():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, password FROM users")
    users = cursor.fetchall()
    for user in users:
        plaintext = user['password']
        # Only hash if not already hashed (bcrypt hashes start with $2)
        if not plaintext.startswith('$2'):
            hashed = bcrypt.hashpw(plaintext.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed.decode('utf-8'), user['id']))
    conn.commit()
    cursor.close()
    conn.close()
    print("All passwords have been securely hashed.")

# Execute the import
insert_employee_data('notebooks/extracted-20.csv', 'emp-data/employee_names.csv')

# Uncomment to run password hashing for all users
hash_all_user_passwords()