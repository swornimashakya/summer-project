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
        connection.commit()

        # Read employee CSV file
        with open(extracted_20_data, 'r') as file:
            csv_reader = csv.DictReader(file)
            employee_id = 1  # Start employee_id from 1
            for row in csv_reader:
                # Get name from the names list, cycling through if necessary
                name = names[(employee_id - 1) % len(names)]
                
                # Calculate fields
                age = int(row['Age'])
                dob = calculate_dob(age)
                years_at_company = int(row['YearsAtCompany'])
                date_joined = calculate_date_joined(years_at_company)
                
                # Insert data into the database
                cursor.execute("""
                    INSERT INTO employees (
                        employee_id,
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    employee_id,                     # employee_id (start from 1)
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
                    1 if row['Attrition'] == 'Yes' else 0  # attrition_risk
                ))
                employee_id += 1
        # Commit the transaction
        connection.commit()
        print("Data inserted successfully!")
    except Error as e:
        print(f"Error: {e}")
    
    cursor.close()
    connection.close()

# Insert to users table
def insertToUsersTable():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # Clear users table
        cursor.execute("DELETE FROM users")
        connection.commit()

        # Fetch all employee IDs and names
        cursor.execute("SELECT employee_id, name FROM employees")
        employees = cursor.fetchall()

        # Insert a user for each employee
        for emp_id, name in employees:
            email = f"{name.lower().replace(' ', '')}@company.com"
            raw_password = "default123"
            hashed_password = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            role = 'employee'
            cursor.execute("""
                INSERT INTO users (employee_id, email, password, role)
                VALUES (%s, %s, %s, %s)
            """, (emp_id, email, hashed_password, role))

        # Add admin user manually (if needed)
        hr_password = bcrypt.hashpw("hr12345".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("""
            INSERT INTO users (employee_id, email, password, role)
            VALUES (%s, %s, %s, %s)
        """, (1, 'hr@company.com', hr_password, 'hr'))

        connection.commit()
        print("Users inserted and mapped to employees successfully with bcrypt passwords!")
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

# Example usage:
insertToDatabase('notebooks\extracted-20.csv', 'emp-data\employee_names.csv')

# Example usage for inserting users
insertToUsersTable()