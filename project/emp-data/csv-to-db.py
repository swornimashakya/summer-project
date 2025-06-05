import csv
import mysql.connector
from mysql.connector import Error
import random
from datetime import datetime, timedelta
import bcrypt

# Function to establish a connection to the MySQL database
def get_db_connection():
    return mysql.connector.connect(
        user='root',
        password='12345',
        host='localhost',
        database='employee_db',
        auth_plugin='mysql_native_password'
    )

# Function to load names from a CSV file
def load_names(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        return [row[0] for row in csv_reader]

# Function to calculate approximate DOB based on age
def calculate_dob(age):
    current_year = datetime.now().year
    birth_year = current_year - age
    # Create a random date within the birth year
    random_day = random.randint(1, 28)
    random_month = random.randint(1, 12)
    return datetime(birth_year, random_month, random_day).date()

# Function to calculate date_joined based on YearsAtCompany
def calculate_date_joined(years_at_company):
    current_date = datetime.now()
    joined_date = current_date - timedelta(days=years_at_company*365)
    return joined_date.date()

# Function to read CSV file and insert data into MySQL database
def insertToDatabase(extracted_20_data, employee_names):
    try:
        # Load names from the names CSV file
        names = load_names(employee_names)
        
        connection = get_db_connection()
        cursor = connection.cursor()

        # Empty the employees table before inserting new data
        cursor.execute("DELETE FROM employees")
        # Reset AUTO_INCREMENT to 1
        cursor.execute("ALTER TABLE employees AUTO_INCREMENT = 1")
        connection.commit()

        # Insert HR as the first employee (auto-increment employee_id)
        cursor.execute("""
            INSERT INTO employees (
                name, dob, age, position, department, salary,
                total_working_years, years_at_company, date_joined,
                marital_status, gender, edufield, env_satisfaction,
                job_involvement, job_level, job_satisfaction, overtime,
                work_life_balance, distance, attrition_risk
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'Swornima Shakya', datetime(2002, 9, 21).date(), 35, 'HR Manager', 'HR', 0.0,
            15, 10, datetime(2015, 1, 1).date(), 'Single', 'Female', 'Human Resources',
            4, 4, 4, 4, 'No', 4, 1, 0
        ))

        # Get the HR's employee_id (should be 1 if table is empty before insert)
        cursor.execute("SELECT employee_id FROM employees WHERE name = %s AND position = %s", ('Swornima Shakya', 'HR Manager'))
        hr_employee_id = cursor.fetchone()[0]

        # Read employee CSV file
        with open(extracted_20_data, 'r') as file:
            csv_reader = csv.DictReader(file)
            employee_id = hr_employee_id + 1  # Start employee_id from next after HR
            for row in csv_reader:
                # Get name from the names list, cycling through if necessary
                name = names[(employee_id - hr_employee_id - 1) % len(names)]
                
                # Calculate fields
                age = int(row['Age'])
                dob = calculate_dob(age)
                years_at_company = int(row['YearsAtCompany'])
                date_joined = calculate_date_joined(years_at_company)
                
                # Insert data into the database (let employee_id auto-increment)
                cursor.execute("""
                    INSERT INTO employees (
                        name,
                        dob,
                        age, 
                        position, 
                        department, 
                        salary, 
                        total_working_years, 
                        years_at_company,
                        date_joined,
                        marital_status,
                        gender,
                        edufield,
                        env_satisfaction,
                        job_involvement,
                        job_level,
                        job_satisfaction,
                        overtime,
                        work_life_balance,
                        distance,
                        attrition_risk
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    name,                            # name
                    dob,                             # dob
                    age,                             # age
                    row['JobRole'],                  # position
                    row['Department'],               # department
                    float(row['MonthlyIncome']),     # salary
                    int(row['TotalWorkingYears']),   # total_working_years
                    years_at_company,                # years_at_company
                    date_joined,                     # date_joined
                    row['MaritalStatus'],            # marital_status
                    row['Gender'],                   # gender
                    row['EducationField'],           # edufield
                    int(row['EnvironmentSatisfaction']), # env_satisfaction
                    int(row['JobInvolvement']),      # job_involvement
                    int(row['JobLevel']),            # job_level
                    int(row['JobSatisfaction']),     # job_satisfaction
                    row['OverTime'],                 # overtime
                    int(row['WorkLifeBalance']),     # work_life_balance
                    int(row['DistanceFromHome']),    # distance
                    0  # attrition_risk (set to 0 initially)
                ))
                employee_id += 1
        # Commit the transaction
        connection.commit()
        print("Data inserted successfully!")
    except Error as e:
        print(f"Error: {e}")
    
    cursor.close()
    connection.close()

def insertToUsersTable():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Clear users table
        cursor.execute("DELETE FROM users")
        # Reset AUTO_INCREMENT to 1
        cursor.execute("ALTER TABLE users AUTO_INCREMENT = 1")
        connection.commit()

        # Fetch all employee IDs and names
        cursor.execute("SELECT employee_id, name, position FROM employees ORDER BY employee_id ASC")
        employees = cursor.fetchall()

        # Insert a user for each employee
        for emp_id, name, position in employees:
            if position == 'HR Manager' and name == 'Swornima Shakya':
                email = "hr@company.com"
                raw_password = "hr12345"
                role = "hr"
            else:
                email = f"{name.lower().replace(' ', '')}@company.com"
                raw_password = "pass123"
                role = "employee"
            hashed_password = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
                INSERT INTO users (employee_id, email, password, role)
                VALUES (%s, %s, %s, %s)
            """, (emp_id, email, hashed_password, role))

        connection.commit()
        print("Users and HR inserted and mapped to employees successfully with bcrypt passwords!")
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

# Example usage:
insertToDatabase(r'D:\BIM\Summer Project\project\notebooks\extracted-20.csv',
                 r'D:\BIM\Summer Project\project\emp-data\employee_names.csv')

# Example usage for inserting users
insertToUsersTable()