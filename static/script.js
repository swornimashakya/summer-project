// Fetch data from Flask backend API
fetch('/get_data')
.then(response => response.json())
.then(data => {
    const ctx = document.getElementById('employeeChart').getContext('2d');
    const employeeChart = new Chart(ctx, {
        type: 'bar',  // Type of chart: bar chart
        data: {
            labels: data.labels,  // Departments
            datasets: [{
                label: 'Number of Employees',
                data: data.values,  // Employee count per department
                backgroundColor: ['#FF5733', '#33FF57', '#3357FF'],
                borderColor: ['#FF5733', '#33FF57', '#3357FF'],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
});