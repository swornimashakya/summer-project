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
    green: '#34A853',
    gray: '#888'
};

// Helper: sort labels/data descending by data value
function sortLabelsData(labels, data) {
    const zipped = labels.map((label, i) => ({ label, value: data[i] }));
    zipped.sort((a, b) => b.value - a.value);
    return {
        labels: zipped.map(z => z.label),
        data: zipped.map(z => z.value)
    };
}

// Helper: get color palette for n items
function getPalette(n, baseColors) {
    const palette = [];
    for (let i = 0; i < n; i++) {
        palette.push(baseColors[i % baseColors.length]);
    }
    return palette;
}

// Chart.js plugin for data labels
// const dataLabelPlugin = {
//     id: 'datalabels',
//     afterDatasetsDraw(chart) {
//         const { ctx } = chart;
//         chart.data.datasets.forEach((dataset, i) => {
//             const meta = chart.getDatasetMeta(i);
//             meta.data.forEach((bar, idx) => {
//                 const value = dataset.data[idx];
//                 if (value > 0) {
//                     ctx.save();
//                     ctx.font = 'bold 11px sans-serif';
//                     ctx.fillStyle = '#222';
//                     ctx.textAlign = 'center';
//                     ctx.textBaseline = 'bottom';
//                     let x = bar.x, y = bar.y;
//                     if (chart.config.type === 'bar') {
//                         y -= 4;
//                     } else if (chart.config.type === 'doughnut') {
//                         const model = bar;
//                         x = model.x;
//                         y = model.y;
//                     }
//                     ctx.fillText(value, x, y);
//                     ctx.restore();
//                 }
//             });
//         });
//     }
// };

let deptChartInstance = null;
function renderDeptChart(labels, data, type, options = {}) {
    const deptCanvas = document.getElementById('deptChart');
    if (!deptCanvas) return;
    const ctx = deptCanvas.getContext('2d');
    if (deptChartInstance) deptChartInstance.destroy();

    // Sort for bar/line
    let sorted = { labels, data };
    if (type === 'bar' || type === 'line') {
        sorted = sortLabelsData(labels, data);
    }

    // Palette
    const baseColors = [colors.blue, colors.red, colors.yellow, colors.green, colors.gray, '#8e44ad', '#16a085', '#e67e22'];
    const chartColors = getPalette(sorted.labels.length, baseColors);

    deptChartInstance = new Chart(ctx, {
        type: type || 'doughnut',
        data: {
            labels: sorted.labels,
            datasets: [{
                label: options.label || '',
                data: sorted.data,
                backgroundColor: chartColors,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: type === 'doughnut'
                    ? { position: 'bottom', labels: { padding: 10, font: { size: 12 } } }
                    : { display: false },
                title: {
                    display: !!options.label,
                    text: options.label,
                    font: { size: 16 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const val = context.parsed;
                            const pct = total ? ((val / total) * 100).toFixed(1) : 0;
                            return `${val} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: type === 'doughnut' ? '70%' : undefined,
            scales: type === 'bar' || type === 'line'
                ? {
                    y: { beginAtZero: true, grid: { display: true, drawBorder: false } },
                    x: { grid: { display: false } }
                }
                : undefined
        }
        // plugins: [dataLabelPlugin]
    });
}

let attritionChartInstance = null;
function renderAttritionChart(labels, data, type, options = {}) {
    const attritionCanvas = document.getElementById('attritionChart');
    if (!attritionCanvas) return;
    const ctx = attritionCanvas.getContext('2d');
    if (attritionChartInstance) attritionChartInstance.destroy();

    // Sort for bar/line
    let sorted = { labels, data };
    if (type === 'bar' || type === 'line') {
        sorted = sortLabelsData(labels, data);
    }

    // Palette
    const baseColors = [colors.red, colors.blue, colors.yellow, colors.green, colors.gray, '#8e44ad', '#16a085', '#e67e22'];
    const chartColors = getPalette(sorted.labels.length, baseColors);

    attritionChartInstance = new Chart(ctx, {
        type: type || 'bar',
        data: {
            labels: sorted.labels,
            datasets: [{
                label: options.label || 'Attrition',
                data: sorted.data,
                backgroundColor: chartColors,
                borderRadius: 6,
                fill: true,
                tension: 0.4,
                borderColor: colors.red
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: type === 'doughnut'
                    ? { position: 'right', labels: { padding: 20, font: { size: 12 } } }
                    : { display: false },
                title: {
                    display: !!options.label,
                    text: options.label,
                    font: { size: 16 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const val = context.parsed;
                            const pct = total ? ((val / total) * 100).toFixed(1) : 0;
                            return `${val} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: type === 'doughnut' ? '70%' : undefined,
            scales: type === 'bar' || type === 'line'
                ? {
                    y: { beginAtZero: true, grid: { display: true, drawBorder: false } },
                    x: { grid: { display: false } }
                }
                : undefined
        },
        // plugins: [dataLabelPlugin]
    });
}

document.addEventListener('DOMContentLoaded', function () {
    const chartData = window.dashboardChartData || {};
    const filter = document.getElementById('chartFilter');
    const attritionFilter = document.getElementById('attritionChartFilter');

    // Initial render (department)
    renderDeptChart(chartData.department_labels, chartData.department_data, 'doughnut', { label: 'Employees by Department' });

    // Initial render for attrition chart (gender)
    renderAttritionChart(chartData.attrition_gender_labels, chartData.attrition_gender_data, 'bar', { label: 'Attrition by Gender' });

    filter && filter.addEventListener('change', function () {
        const value = this.value;
        if (value === 'department') {
            renderDeptChart(chartData.department_labels, chartData.department_data, 'doughnut', { label: 'Employees by Department' });
        } else if (value === 'age') {
            renderDeptChart(chartData.age_labels, chartData.age_data, 'bar', { label: 'Employees by Age Group' });
        } else if (value === 'salary') {
            renderDeptChart(chartData.salary_labels, chartData.salary_data, 'line', { label: 'Employees by Salary Band' });
        }
    });

    attritionFilter && attritionFilter.addEventListener('change', function () {
        const value = this.value;
        if (value === 'gender') {
            renderAttritionChart(chartData.attrition_gender_labels, chartData.attrition_gender_data, 'bar', { label: 'Attrition by Gender' });
        } else if (value === 'age') {
            renderAttritionChart(chartData.attrition_age_labels, chartData.attrition_age_data, 'bar', { label: 'Attrition by Age Group' });
        } else if (value === 'department') {
            renderAttritionChart(chartData.attrition_dept_labels, chartData.attrition_dept_data, 'bar', { label: 'Attrition by Department' });
        } else if (value === 'salary') {
            renderAttritionChart(chartData.attrition_salary_labels, chartData.attrition_salary_data, 'line', { label: 'Attrition by Salary Band' });
        } else if (value === 'overtime') {
            renderAttritionChart(chartData.attrition_overtime_labels, chartData.attrition_overtime_data, 'bar', { label: 'Attrition by Overtime' });
        }
    });
});