import React from 'react';

function ControlPanel({ onClick, isWebcamOn }) {
    const buttonText = isWebcamOn ? "Stop" : "Start";

    return (
        <div>
            <button onClick={onClick}>
                {buttonText}
            </button>
        </div>
    );
}

export default ControlPanel;