// Department Chart (Pie)
fetch('/department-data')
    .then(res => res.json())
    .then(data => {
        new Chart(document.getElementById('deptChart'), {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.counts,
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'
                    ]
                }]
            }
        });
    });

// Age Chart (Bar)
fetch('/age-data')
    .then(res => res.json())
    .then(data => {
        new Chart(document.getElementById('ageChart'), {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Employees',
                    data: data.counts,
                    backgroundColor: '#4BC0C0'
                }]
            },
            options: {
                scales: { 
                  y: { 
                    beginAtZero: true,
                    ticks: {
                      stepSize: 2
                    } 
                  }
                }
            }
        });
    });

// Salary Chart (Line)
fetch('/salary-data')
    .then(res => res.json())
    .then(data => {
        new Chart(document.getElementById('salaryChart'), {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Employees',
                    data: data.counts,
                    borderColor: '#FF6384',
                    fill: false
                }]
            },
            options: {
                scales: { y: { beginAtZero: true } }
            }
        });
    });