// Listens for the extension icon click to open the side panel
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});

// The core listener for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'rewriteText' && message.text) {
    const { text, history } = message;

    console.log("Background script received:", { text, history });

    // Call our local backend API
    fetch('http://127.0.0.1:5000/api/rewrite', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // Send both the current query and the conversation history
      body: JSON.stringify({ text: text, history: history }),
    })
    .then(response => {
        if (!response.ok) {
            // Handle HTTP errors like 500
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
      // Send the rewritten text back to the content script that requested it
      sendResponse({ status: 'success', data: data });

      // Also, send the result to the side panel so it can display it
      chrome.runtime.sendMessage({
          type: 'displayResult',
          original: text,
          rewritten: data.rewritten_text
      });
    })
    .catch(error => {
      console.error('Error calling backend:', error);
      sendResponse({ status: 'error', error: error.message });
    });

    // Return true to indicate that we will send a response asynchronously
    return true;
  }
});