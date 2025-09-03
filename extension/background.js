// extension/background.js Listens for the extension icon click to open the side panel
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});

// The core listener for messages from the extension
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  
  // --- New Handler for the full chat UI in the side panel ---
  if (message.type === 'getChatResponse') {
    const { query, history } = message;
    fetch('http://127.0.0.1:5000/api/chat', { // Using a new, more descriptive endpoint
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: query, history: history }),
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
    })
    .then(data => sendResponse({ status: 'success', data: data }))
    .catch(error => sendResponse({ status: 'error', error: error.message }));
    
    return true; // Indicate async response
  }

  // --- Kept for the inline "Rewrite" functionality from content.js ---
  if (message.type === 'rewriteText') {
    const { text } = message;
    fetch('http://127.0.0.1:5000/api/chat', { // Can reuse the same endpoint
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, history: [] }), // Inline rewrite has no history
    })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return response.json();
    })
    .then(data => sendResponse({ status: 'success', data: data }))
    .catch(error => sendResponse({ status: 'error', error: error.message }));
    
    return true; // Indicate async response
  }
});