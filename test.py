import mysql.connector

print("Connecting to MySQL database...")
connection = mysql.connector.connect(
    host='localhost',         # Your MySQL host (usually localhost)
    user='root',              # Your MySQL username
    password='',         # Your MySQL root password
    database='information_schema'    # Your database name (adjust it if it's different)
)

print("Connection successful")