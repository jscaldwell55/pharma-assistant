let rewriteButton = null;
let currentSelection = null;

// Array to store the chat history for the current page session
let chatHistory = [];

// Function to create and show the rewrite button
function showRewriteButton(x, y) {
    if (!rewriteButton) {
        rewriteButton = document.createElement('button');
        rewriteButton.innerHTML = '✨ Rewrite';
        // Style the button
        Object.assign(rewriteButton.style, {
            position: 'absolute',
            zIndex: '99999',
            padding: '8px 12px',
            backgroundColor: '#0A9A78',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
            fontSize: '14px',
            fontFamily: 'sans-serif'
        });
        document.body.appendChild(rewriteButton);
        rewriteButton.addEventListener('click', handleRewriteClick);
    }
    rewriteButton.style.left = `${x}px`;
    rewriteButton.style.top = `${y}px`;
    rewriteButton.style.display = 'block';
}

// Function to hide the button
function hideRewriteButton() {
    if (rewriteButton) {
        rewriteButton.style.display = 'none';
    }
}

// Event handler for when the user selects text
document.addEventListener('mouseup', (event) => {
    // A small delay to ensure the selection is finalized
    setTimeout(() => {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();

        if (selectedText.length > 5) {
            currentSelection = selection.getRangeAt(0);
            const rect = currentSelection.getBoundingClientRect();
            showRewriteButton(rect.left + window.scrollX, rect.bottom + window.scrollY + 5);
        } else {
            hideRewriteButton();
        }
    }, 10);
});

// Hide button if user clicks elsewhere on the page
document.addEventListener('mousedown', (event) => {
    if (rewriteButton && event.target !== rewriteButton) {
        hideRewriteButton();
    }
});

// Function to handle the button click
function handleRewriteClick() {
    if (!currentSelection) return;
    const textToRewrite = currentSelection.toString();

    rewriteButton.textContent = '...';
    rewriteButton.disabled = true;

    // Add the user's message to our history before sending
    chatHistory.push({ role: 'user', content: textToRewrite });

    chrome.runtime.sendMessage(
        { 
            type: 'rewriteText', 
            text: textToRewrite,
            history: chatHistory // Pass the current history
        }, 
        (response) => {
            if (response && response.status === 'success' && response.data.rewritten_text) {
                const rewrittenText = response.data.rewritten_text;
                
                // Replace the original selection with the new text
                currentSelection.deleteContents();
                currentSelection.insertNode(document.createTextNode(rewrittenText));

                // Add the assistant's response to our history
                chatHistory.push({ role: 'assistant', content: rewrittenText });

            } else {
                console.error("Rewrite failed:", response ? response.error : 'No response');
                alert('Rewrite failed. Check the extension console and backend server for errors.');
                // If the request fails, remove the last user message from history to keep it clean
                chatHistory.pop();
            }

            // Reset the button state and hide it
            rewriteButton.textContent = '✨ Rewrite';
            rewriteButton.disabled = false;
            hideRewriteButton();
        }
    );
}