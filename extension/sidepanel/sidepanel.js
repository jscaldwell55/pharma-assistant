// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'displayResult') {
        const container = document.getElementById('last-query');
        // Update the content with the latest query and response
        container.innerHTML = `
            <strong>Original Query:</strong>
            <pre>${escapeHtml(message.original)}</pre>
            <strong>Assistant Response:</strong>
            <pre>${escapeHtml(message.rewritten)}</pre>
        `;
    }
});

// Simple helper to prevent HTML injection in the side panel
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') {
        return '';
    }
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }