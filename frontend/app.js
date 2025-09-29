// API Configuration - will be automatically updated based on deployment
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8080'
    : 'https://pharma-assistant-api-42246411579.us-central1.run.app';

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');

// Store conversation history
let conversationHistory = [];

// Send message function
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Disable input
    userInput.disabled = true;
    sendButton.disabled = true;
    errorMessage.classList.remove('active');

    // Add user message to UI
    addMessageToChat('user', message);
    
    // Clear input
    userInput.value = '';

    // Show loading
    loading.classList.add('active');

    try {
        // Make API call
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: message,
                history: conversationHistory,
                include_trace: false  // Set to true for debugging
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // Add assistant response to UI
        addMessageToChat('assistant', data.response_text, data);
        
        // Update conversation history
        conversationHistory.push(
            { role: 'user', content: message },
            { role: 'assistant', content: data.response_text }
        );
        
        // Keep history limited to last 10 exchanges
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Sorry, there was an error processing your request. Please try again.');
    } finally {
        // Hide loading
        loading.classList.remove('active');
        
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

// Add message to chat UI
function addMessageToChat(role, content, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Parse content for safety notes
    const safetyNotePattern = /\*\*Safety note:\*\* (.+)/;
    const match = content.match(safetyNotePattern);
    
    if (match) {
        // Split content and safety note
        const mainContent = content.replace(match[0], '').trim();
        contentDiv.innerHTML = formatMessage(mainContent);
        
        // Add safety note
        const safetyNote = document.createElement('div');
        safetyNote.className = 'safety-note';
        safetyNote.textContent = match[1];
        contentDiv.appendChild(safetyNote);
    } else {
        contentDiv.innerHTML = formatMessage(content);
    }
    
    // Add metadata if assistant message with data
    if (role === 'assistant' && metadata && metadata.safety_labels) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'metadata';
        
        const labels = metadata.safety_labels;
        const relevantLabels = [];
        
        if (labels.grounded) {
            relevantLabels.push('<span class="metadata-item"><span class="metadata-label">✓</span> Grounded</span>');
        }
        if (labels.medical_advice) {
            relevantLabels.push('<span class="metadata-item"><span class="metadata-label">⚕</span> Medical Advice Detected</span>');
        }
        if (labels.adverse_event) {
            relevantLabels.push('<span class="metadata-item"><span class="metadata-label">⚠</span> AE Report</span>');
        }
        
        if (relevantLabels.length > 0) {
            metadataDiv.innerHTML = relevantLabels.join('');
            contentDiv.appendChild(metadataDiv);
        }
    }
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format message content
function formatMessage(text) {
    // Convert markdown-like formatting
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\n\n/g, '</p><p>');
    text = text.replace(/\n/g, '<br>');
    
    // Handle bullet points
    text = text.replace(/^- (.+)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Wrap in paragraph tags if not already
    if (!text.startsWith('<p>') && !text.startsWith('<ul>')) {
        text = '<p>' + text + '</p>';
    }
    
    return text;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
    setTimeout(() => {
        errorMessage.classList.remove('active');
    }, 5000);
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Focus input on load
userInput.focus();

// Initialize with warmup call
window.addEventListener('load', async () => {
    try {
        // Warm up the API on page load
        await fetch(`${API_BASE_URL}/api/warmup`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        console.log('API warmed up successfully');
    } catch (error) {
        console.warn('Could not warm up API:', error);
    }
});