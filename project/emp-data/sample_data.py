import mysql.connector
import random
from datetime import datetime, timedelta

def load_name_list(filename='project\employee_names.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="employee_db",
        auth_plugin='mysql_native_password'
    )

def insert_sample_employees():
    conn = create_connection()
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM employees")

    # Reset ID counter
    cursor.execute("ALTER TABLE employees AUTO_INCREMENT = 1")
    
    # Load your name list
    try:
        names = load_name_list()
        if not names:
            raise ValueError("No names found in the file")
    except FileNotFoundError:
        print("Name file not found, using generated names")
        names = [f"Employee {i}" for i in range(1, 51)]

    # Tech company departments and positions
    departments = ['Engineering', 'Product', 'Sales', 'People', 'R&D']
    positions = {
        'Engineering': ['Machine Learning Engineer', 'Data Scientist', 
                       'Software Engineer', 'DevOps Engineer'],
        'Product': ['Product Manager', 'UX Designer', 'Product Designer'],
        'R&D': ['AI Researcher', 'Research Scientist'],
        'Sales': ['Account Executive', 'Sales Development Rep', 
                  'Sales Manager'],
        'People': ['HR Manager', 'Talent Acquisition']
    }
    
    # Generate employees using the name list
    for i, name in enumerate(names[:20], 1):
        age = random.randint(18, 60)
        department = random.choice(departments)
        position = random.choice(positions[department])
        salary = random.randint(20000, 150000)  # Need to normalize this
        status = random.choice(['Present', 'Absent', 'Left'])
        total_working_years = random.randint(1, 40)
        years_at_company = random.randint(1, total_working_years)
        job_satisfaction = random.randint(1, 4)
        overtime = random.choice(['Yes', 'No'])
        marital_status = random.choice(['Single', 'Married', 'Divorced'])
        gender = random.choice(['Male', 'Female'])
        
        # Create a random join date
        join_date = datetime.now() - timedelta(days=365*years_at_company)
        
        cursor.execute(
            """INSERT INTO employees 
            (name, age, position, department, salary, status, 
             total_working_years, years_at_company, job_satisfaction, 
             overtime, marital_status, gender, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (name, age, position, department, salary, status,
             total_working_years, years_at_company, job_satisfaction,
             overtime, marital_status, gender, join_date)
        )
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Inserted 50 tech employee samples")

if __name__ == "__main__":
    insert_sample_employees()