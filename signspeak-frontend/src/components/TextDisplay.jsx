import React from 'react';

/**
 * Displays the translated text received from the backend.
 * @param {string} text - the text to display.
 */
function TextDisplay({ text }) {
    return (
        <div className="text-display-container">
            <div className="text-display-box">
                {text ? text : <span className="placeholder">Translated text will appear here...</span>}
            </div>
        </div>
    );
}

export default TextDisplay;