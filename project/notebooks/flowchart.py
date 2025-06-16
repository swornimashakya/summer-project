from graphviz import Digraph

def make_flowchart(title, steps):
    dot = Digraph(comment=title)
    dot.attr(rankdir='TB', size='8')

    # Only process steps that have an 'id' key
    for step in steps:
        if 'id' in step:
            id = step['id']
            label = step['label']
            shape = step.get('shape', 'box')  # default is process
            dot.node(id, label, shape=shape)

    for src, dst, label in steps[-1].get('edges', []):
        dot.edge(src, dst, label=label)

    return dot

# 🔐 1. User Authentication
auth_steps = [
    {"id": "start", "label": "Start", "shape": "ellipse"},
    {"id": "login", "label": "Display Login Form", "shape": "parallelogram"},
    {"id": "input", "label": "Input Email & Password", "shape": "parallelogram"},
    {"id": "query", "label": "Query Users table", "shape": "box"},
    {"id": "check_user", "label": "User Found?", "shape": "diamond"},
    {"id": "invalid_user", "label": "Show 'Invalid Email'", "shape": "box"},
    {"id": "check_pass", "label": "Correct Password?", "shape": "diamond"},
    {"id": "invalid_pass", "label": "Show 'Invalid Password'", "shape": "box"},
    {"id": "role", "label": "Set Session & Role", "shape": "box"},
    {"id": "hr", "label": "Go to HR Dashboard", "shape": "box"},
    {"id": "emp", "label": "Go to Employee View", "shape": "box"},
    {"id": "end", "label": "End", "shape": "ellipse"},
    {"edges": [
        ("start", "login", ""),
        ("login", "input", ""),
        ("input", "query", ""),
        ("query", "check_user", ""),
        ("check_user", "invalid_user", "No"),
        ("invalid_user", "end", ""),
        ("check_user", "check_pass", "Yes"),
        ("check_pass", "invalid_pass", "No"),
        ("invalid_pass", "end", ""),
        ("check_pass", "role", "Yes"),
        ("role", "hr", "If HR"),
        ("role", "emp", "If Employee"),
        ("hr", "end", ""),
        ("emp", "end", "")
    ]}
]

# 📝 2. Job Application
job_steps = [
    {"id": "start", "label": "Start", "shape": "ellipse"},
    {"id": "form", "label": "Show Application Form", "shape": "parallelogram"},
    {"id": "input", "label": "Input Applicant Info", "shape": "parallelogram"},
    {"id": "validate", "label": "Validate Input", "shape": "box"},
    {"id": "insert", "label": "Insert into Applicants", "shape": "box"},
    {"id": "msg", "label": "Show Confirmation", "shape": "box"},
    {"id": "end", "label": "End", "shape": "ellipse"},
    {"edges": [
        ("start", "form", ""),
        ("form", "input", ""),
        ("input", "validate", ""),
        ("validate", "insert", "Valid"),
        ("insert", "msg", ""),
        ("msg", "end", "")
    ]}
]

# 🧑‍💼 3. Employee Data Management (Add/Edit Only)
emp_steps = [
    {"id": "start", "label": "Start", "shape": "ellipse"},
    {"id": "login", "label": "HR Logs In", "shape": "box"},
    {"id": "choose", "label": "Choose Action (Add/Edit)", "shape": "diamond"},
    {"id": "add", "label": "Input New Employee Data", "shape": "parallelogram"},
    {"id": "insert", "label": "Insert into Employees & Users", "shape": "box"},
    {"id": "predict_add", "label": "Predict Attrition", "shape": "box"},
    {"id": "edit", "label": "Update Employee Data", "shape": "parallelogram"},
    {"id": "update", "label": "Update DB & Predict", "shape": "box"},
    {"id": "end", "label": "End", "shape": "ellipse"},
    {"edges": [
        ("start", "login", ""),
        ("login", "choose", ""),
        ("choose", "add", "Add"),
        ("add", "insert", ""),
        ("insert", "predict_add", ""),
        ("predict_add", "end", ""),
        ("choose", "edit", "Edit"),
        ("edit", "update", ""),
        ("update", "end", "")
    ]}
]

# 📅 4. Leave Management
leave_steps = [
    {"id": "start", "label": "Start", "shape": "ellipse"},
    {"id": "emp_login", "label": "Employee Logs In", "shape": "box"},
    {"id": "leave_form", "label": "Fill Leave Form", "shape": "parallelogram"},
    {"id": "validate", "label": "Validate Dates", "shape": "diamond"},
    {"id": "save", "label": "Insert into Leave Requests", "shape": "box"},
    {"id": "confirm", "label": "Show Confirmation", "shape": "box"},
    {"id": "end", "label": "End", "shape": "ellipse"},
    {"edges": [
        ("start", "emp_login", ""),
        ("emp_login", "leave_form", ""),
        ("leave_form", "validate", ""),
        ("validate", "save", "Valid"),
        ("save", "confirm", ""),
        ("confirm", "end", "")
    ]}
]

# 🔮 5. Churn Prediction
churn_steps = [
    {"id": "start", "label": "Start", "shape": "ellipse"},
    {"id": "hr_login", "label": "HR Logs In", "shape": "box"},
    {"id": "trigger", "label": "Click Predict Button", "shape": "box"},
    {"id": "fetch", "label": "Fetch Employees", "shape": "box"},
    {"id": "loop", "label": "For Each Employee", "shape": "box"},
    {"id": "preprocess", "label": "Preprocess Data", "shape": "box"},
    {"id": "predict", "label": "Run ML Model", "shape": "box"},
    {"id": "update", "label": "Update attrition_risk", "shape": "box"},
    {"id": "show", "label": "Show At-Risk Employees", "shape": "box"},
    {"id": "end", "label": "End", "shape": "ellipse"},
    {"edges": [
        ("start", "hr_login", ""),
        ("hr_login", "trigger", ""),
        ("trigger", "fetch", ""),
        ("fetch", "loop", ""),
        ("loop", "preprocess", ""),
        ("preprocess", "predict", ""),
        ("predict", "update", ""),
        ("update", "show", ""),
        ("show", "end", "")
    ]}
]

# Generate Diagrams
make_flowchart("User Authentication", auth_steps).render('flowchart_auth', format='png', cleanup=True)
make_flowchart("Job Application", job_steps).render('flowchart_job', format='png', cleanup=True)
make_flowchart("Employee Management", emp_steps).render('flowchart_emp', format='png', cleanup=True)
make_flowchart("Leave Management", leave_steps).render('flowchart_leave', format='png', cleanup=True)
make_flowchart("Churn Prediction", churn_steps).render('flowchart_churn', format='png', cleanup=True)
