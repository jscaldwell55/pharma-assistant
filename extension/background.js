// extension/background.js
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});

// Configure API base URL - use localhost if available, otherwise production
const API_BASE_URL = 'https://pharma-assistant-api.onrender.com';
const LOCAL_API_URL = 'http://127.0.0.1:5000';

// Helper function to determine which URL to use
async function getApiUrl() {
  try {
    // Try local first (for development)
    const response = await fetch(`${LOCAL_API_URL}/api/health`, { 
      method: 'GET',
      signal: AbortSignal.timeout(2000) // 2 second timeout
    });
    if (response.ok) {
      console.log('Using local development server');
      return LOCAL_API_URL;
    }
  } catch (error) {
    console.log('Local server not available, using production');
  }
  return API_BASE_URL;
}

// The core listener for messages from the extension
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  
  // --- New Handler for the full chat UI in the side panel ---
  if (message.type === 'getChatResponse') {
    const { query, history } = message;
    
    getApiUrl().then(apiUrl => {
      return fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query, history: history }),
      });
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
    
    getApiUrl().then(apiUrl => {
      return fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, history: [] }), // Inline rewrite has no history
      });
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