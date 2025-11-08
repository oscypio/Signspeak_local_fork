import React from 'react';

/**
 * Renders the Start/Stop button.
 * @param {function} onClick - The toggleWebcam function.
 * @param {boolean} isWebcamOn - Whether the webcam is currently active.
 * @param {string} status - The current application status (IDLE, CONNECTING, etc.)
 */
function ControlPanel({ onClick, isWebcamOn, status }) {
    const buttonText = isWebcamOn ? "Stop" : "Start";

    // Disable the button while connecting or processing the final translation
    const isDisabled = (status === 'CONNECTING' || status === 'PROCESSING');

    return (
        <div>
            <button onClick={onClick} disabled={isDisabled}>
                {buttonText}
            </button>
        </div>
    );
}

export default ControlPanel;