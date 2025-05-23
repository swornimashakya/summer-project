import mysql.connector
import bcrypt

def get_db_connection():
    return mysql.connector.connect(
        user='root',
        password='12345',
        host='localhost',
        database='employee_db',
        auth_plugin='mysql_native_password'
    )

connection = get_db_connection()
cursor = connection.cursor(dictionary=True)

cursor.execute("SELECT id, password FROM users")
users = cursor.fetchall()

for user in users:
    plaintext = user['password']
    hashed = bcrypt.hashpw(plaintext.encode('utf-8'), bcrypt.gensalt())
    
    cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed.decode('utf-8'), user['id']))

connection.commit()
cursor.close()
connection.close()

print("All passwords have been securely hashed.")
