document.getElementById('upload-form').onsubmit = async function (event) {
    event.preventDefault(); // Prevent form submission

    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    
    // Check if a file has been selected
    if (fileInput.files.length === 0) {
        document.getElementById('response').innerText = 'Please select a file.';
        return;
    }
    
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const csv = data.csv;

        // Parse the CSV data and create a table
        const rows = csv.split('\n').map(row => row.split(','));
        createTable(rows);
    } catch (error) {
        console.error('Error uploading file:', error);
        document.getElementById('response').innerText = 'Error uploading file: ' + error.message;
    }
};

function createTable(rows) {
    const table = document.createElement('table');
    table.setAttribute('border', '1'); // Add a border for better visibility
    const tbody = document.createElement('tbody');

    rows.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        
        row.forEach((cell, cellIndex) => {
            const td = document.createElement('td');
            td.textContent = cell; // Populate cell with data
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = ''; // Clear previous response
    responseDiv.appendChild(table); // Append the new table
}

// Handle CSV download
document.getElementById('download-csv').onclick = function() {
    const rows = document.querySelectorAll('table tr');
    if (rows.length === 0) {
        alert('No data available for download.');
        return;
    }

    const csvContent = Array.from(rows)
        .map(row => Array.from(row.cells).map(cell => cell.textContent).join(','))
        .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'data.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link); // Clean up
};

document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Handle file upload and parsing here
    // After parsing and generating the predicted values:
    document.getElementById('download-csv').style.display = 'block';
});

let toggleForm = () => {
    let signIn = document.querySelector('.sign-in');
    let signUp = document.querySelector('.sign-up');

    if (signIn.style.display === 'none') {
        signIn.style.display = 'block';
        signUp.style.display = 'none';
    } else {
        signIn.style.display = 'none';
        signUp.style.display = 'block';
    }
};

document.querySelector('.sign-up a').addEventListener('click', toggleForm);
document.querySelector('.sign-in a').addEventListener('click', toggleForm);
