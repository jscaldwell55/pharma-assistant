// extension/content.js
let rewriteButton = null;
let currentSelection = null;
// No chatHistory here; it's managed by the side panel. This script is for stateless inline actions.

// ... (showRewriteButton, hideRewriteButton, mouseup/mousedown listeners are the same) ...

function handleRewriteClick() {
    if (!currentSelection) return;
    const textToRewrite = currentSelection.toString();

    rewriteButton.textContent = '...';
    rewriteButton.disabled = true;

    // Call the same message type as the side panel, but with an empty history
    chrome.runtime.sendMessage(
        { 
            type: 'getChatResponse', // Use the main chat handler
            query: textToRewrite,
            history: [] // Inline actions are stateless
        }, 
        (response) => {
            if (response && response.status === 'success' && response.data.response_text) {
                const rewrittenText = response.data.response_text;
                currentSelection.deleteContents();
                currentSelection.insertNode(document.createTextNode(rewrittenText));
            } else {
                console.error("Rewrite failed:", response ? response.error : 'No response');
                alert('Rewrite failed. Check the extension console and backend server for errors.');
            }
            rewriteButton.textContent = '✨ Rewrite';
            rewriteButton.disabled = false;
            hideRewriteButton();
        }
    );
}

// --- Functions to be copy-pasted into content.js ---
function showRewriteButton(x, y) {
    if (!rewriteButton) {
        rewriteButton = document.createElement('button');
        rewriteButton.innerHTML = '✨ Rewrite';
        Object.assign(rewriteButton.style, {
            position: 'absolute', zIndex: '99999', padding: '8px 12px',
            backgroundColor: '#0A9A78', color: 'white', border: 'none',
            borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
            fontSize: '14px', fontFamily: 'sans-serif'
        });
        document.body.appendChild(rewriteButton);
        rewriteButton.addEventListener('click', handleRewriteClick);
    }
    rewriteButton.style.left = `${x}px`;
    rewriteButton.style.top = `${y}px`;
    rewriteButton.style.display = 'block';
}

function hideRewriteButton() { if (rewriteButton) rewriteButton.style.display = 'none'; }
document.addEventListener('mouseup', (event) => { setTimeout(() => { const selection = window.getSelection(); const selectedText = selection.toString().trim(); if (selectedText.length > 5) { currentSelection = selection.getRangeAt(0); const rect = currentSelection.getBoundingClientRect(); showRewriteButton(rect.left + window.scrollX, rect.bottom + window.scrollY + 5); } else { hideRewriteButton(); } }, 10); });
document.addEventListener('mousedown', (event) => { if (rewriteButton && event.target !== rewriteButton) { hideRewriteButton(); } });