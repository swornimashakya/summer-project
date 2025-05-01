from flask import Flask, render_template, jsonify
import random # For simulating data, remove in production

app = Flask(__name__)

# Database connection function
# def get_db_connection():
#     connection = mysql.connector.connect(
#         host='localhost',         # Your MySQL host (usually localhost)
#         user='root',              # Your MySQL username
#         password='',              # Your MySQL root password
#         database='employee_db',    # Your database name (adjust it if it's different)
#         port=3306           # Your MySQL port (default is 3306)
#     )
#     return connection

@app.route('/')
def index():

    # Simulate a random attrition rate (replace with actual model prediction logic)
    attrition_rate = random.uniform(10, 30)  # Just a random number for now
    return render_template('index.html', attrition_rate=attrition_rate)

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
