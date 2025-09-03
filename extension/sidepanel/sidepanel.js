// --- extension/sidepanel/sidepanel.js DOM Elements ---
const chatContainer = document.getElementById('chat-container');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const newConversationBtn = document.getElementById('new-conversation-btn');
const showTraceCheckbox = document.getElementById('show-trace-checkbox');
const thinkingIndicator = document.getElementById('thinking-indicator');
// The "Clear Cache" button from the old app.py maps to clearing the *backend* cache, not the extension's.
// We are removing it from this UI to avoid confusion, as clearing the backend is a developer action.
// The "New Conversation" button provides the user-facing cache clearing they expect.


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

    // Send message to background script for processing
    chrome.runtime.sendMessage(
        { type: 'getChatResponse', query: userQuery, history: chatHistory },
        (response) => {
            thinkingIndicator.classList.add('hidden');

            if (response && response.status === 'success') {
                const assistantResponse = {
                    role: 'assistant',
                    content: response.data.response_text,
                    safety_labels: response.data.safety_labels,
                    trace: response.data.trace
                };
                chatHistory.push(assistantResponse);
            } else {
                const errorResponse = {
                    role: 'assistant',
                    content: `Sorry, an error occurred. ${response?.error || 'Please check the console.'}`
                };
                chatHistory.push(errorResponse);
            }
            renderMessages();
        }
    );
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

// --- Event Listeners ---
chatForm.addEventListener('submit', handleFormSubmit);
newConversationBtn.addEventListener('click', startNewConversation);
showTraceCheckbox.addEventListener('change', toggleTraceVisibility);

// --- Initial Render ---
startNewConversation();