// --- DOM Elements ---
const chatContainer = document.getElementById('chat-container');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const newConversationBtn = document.getElementById('new-conversation-btn');
const clearCacheBtn = document.getElementById('clear-cache-btn');
const showTraceCheckbox = document.getElementById('show-trace-checkbox');
const thinkingIndicator = document.getElementById('thinking-indicator');

// --- State Management ---
let chatHistory = [];

// --- Functions ---

/**
 * Creates an HTML element for a single chat message.
 * @param {object} message - The message object { role, content, safety_labels, trace }
 * @returns {HTMLElement} The created message element.
 */
function createMessageElement(message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', message.role);

    // Use <p> tags and replace newlines with <br> for basic formatting
    const contentP = document.createElement('p');
    contentP.innerHTML = message.content.replace(/\n/g, '<br>');
    messageDiv.appendChild(contentP);

    // Add safety labels caption if it's an assistant message
    if (message.role === 'assistant' && message.safety_labels) {
        const labels = message.safety_labels;
        const captionText = `Safety · grounded: ${labels.grounded} · med_advice: ${labels.medical_advice} · off_label: ${labels.off_label_use}`;
        const caption = document.createElement('p');
        caption.className = 'safety-caption';
        caption.textContent = captionText;
        messageDiv.appendChild(caption);
    }

    // Add trace expander if it's an assistant message with a trace
    if (message.role === 'assistant' && message.trace) {
        const traceContainer = document.createElement('div');
        traceContainer.className = 'trace-container';
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Show Trace JSON';
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(message.trace, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        traceContainer.appendChild(details);
        messageDiv.appendChild(traceContainer);
    }
    
    return messageDiv;
}

/**
 * Renders the entire chat history to the screen.
 */
function renderMessages() {
    chatContainer.innerHTML = ''; // Clear existing messages
    chatHistory.forEach(message => {
        const messageElement = createMessageElement(message);
        chatContainer.appendChild(messageElement);
    });
    // Scroll to the latest message
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Handles the chat form submission.
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    const userQuery = chatInput.value.trim();
    if (!userQuery) return;

    chatHistory.push({ role: 'user', content: userQuery });
    renderMessages();
    chatForm.reset();
    thinkingIndicator.classList.remove('hidden');

    try {
        // Send request to Flask API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: userQuery,
                history: chatHistory
            })
        });

        thinkingIndicator.classList.add('hidden');

        if (response.ok) {
            const data = await response.json();
            const assistantResponse = {
                role: 'assistant',
                content: data.response_text,
                safety_labels: data.safety_labels,
                trace: data.trace
            };
            chatHistory.push(assistantResponse);
        } else {
            const errorResponse = {
                role: 'assistant',
                content: `Sorry, an error occurred. Status: ${response.status}`
            };
            chatHistory.push(errorResponse);
        }
    } catch (error) {
        thinkingIndicator.classList.add('hidden');
        const errorResponse = {
            role: 'assistant',
            content: `Sorry, a network error occurred: ${error.message}`
        };
        chatHistory.push(errorResponse);
    }

    renderMessages();
}

/**
 * Clears the chat history and starts a new conversation.
 */
function startNewConversation() {
    chatHistory = [];
    renderMessages();
}

/**
 * Toggles the visibility of trace containers based on checkbox state.
 */
function toggleTraceVisibility() {
    document.body.classList.toggle('show-traces', showTraceCheckbox.checked);
}

/**
 * Clears cache and restarts the conversation.
 */
function clearCache() {
    // Clear local storage (if any)
    localStorage.clear();
    sessionStorage.clear();
    
    // Clear chat history
    chatHistory = [];
    renderMessages();
    
    // Optional: You could add a call to server to clear server-side cache
    // For now, just simulate cache clearing with page reload
    alert('Cache cleared! Starting fresh conversation.');
}

// --- Event Listeners ---
chatForm.addEventListener('submit', handleFormSubmit);
newConversationBtn.addEventListener('click', startNewConversation);
clearCacheBtn.addEventListener('click', clearCache);
showTraceCheckbox.addEventListener('change', toggleTraceVisibility);

// --- Initial Render ---
startNewConversation();