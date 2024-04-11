function sendMessage() {
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const userText = userInput.value;

    // Display user message
    const userMessage = document.createElement('div');
    userMessage.textContent = "You: " + userText;
    chatBox.appendChild(userMessage);

    // Send to backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userText }),
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        const botResponse = document.createElement('div');
        botResponse.textContent = "AyurBot: " + data.response;
        chatBox.appendChild(botResponse);
    })
    .catch((error) => {
        console.error('Error:', error);
    });

    userInput.value = ''; // Clear input after sending
}
