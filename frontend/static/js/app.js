function classifyEmail() {
    const emailContent = document.getElementById('email-content').value;

    fetch('http://localhost:8080/api/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_content: emailContent }),
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `Kết quả: ${data.result}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
