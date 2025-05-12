
// Salary Chart
fetch('/salary-data')
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('salaryChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Employees',
                    data: data.counts,
                    borderColor: '#A5B4FC', // Pastel Indigo
                    backgroundColor: 'rgba(165, 180, 252, 0.1)', // Very light pastel indigo
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointBackgroundColor: '#818CF8', // Darker Indigo
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#E5E7EB' // Light gray grid lines
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    });

// Google's color palette
const colors = {
    blue: '#4285F4',
    red: '#EA4335',
    yellow: '#FBBC05',
    green: '#34A853'
};

// Department Distribution Chart
const deptCtx = document.getElementById('deptChart').getContext('2d');
new Chart(deptCtx, {
    type: 'doughnut',
    data: {
        labels: {{ department_labels|tojson }},
        datasets: [{
            data: {{ department_data|tojson }},
            backgroundColor: [
                colors.blue,
                colors.red,
                colors.yellow,
                colors.green
            ],
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'right',
                labels: {
                    padding: 20,
                    font: {
                        size: 12
                    }
                }
            }
        },
        cutout: '70%'
    }
});

// Age Distribution Chart
const ageCtx = document.getElementById('ageChart').getContext('2d');
new Chart(ageCtx, {
    type: 'bar',
    data: {
        labels: {{ age_labels|tojson }},
        datasets: [{
            label: 'Number of Employees',
            data: {{ age_data|tojson }},
            backgroundColor: colors.blue,
            borderRadius: 6
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    display: true,
                    drawBorder: false
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});