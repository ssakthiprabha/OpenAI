document.getElementById('ask-button').onclick = function() {
    const query = document.getElementById('query').value;
    // Send the query to the backend for processing
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: query })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('answer').innerText = data.answer;
    });
};