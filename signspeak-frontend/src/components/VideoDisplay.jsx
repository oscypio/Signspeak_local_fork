import React, { useEffect, forwardRef } from 'react';

/**
 * Shows status (text, spinner) or the video, all within the same frame.
 */
const VideoDisplay = forwardRef(({ status, stream }, ref) => {

    // This hook connects the MediaStream to the <video> tag
    useEffect(() => {
        if (ref.current && stream && status === 'LISTENING') {
            ref.current.srcObject = stream;
        }
    }, [stream, status, ref]);

    /**
     * Renders the correct content based on the application status.
     */
    const renderContent = () => {
        switch (status) {
            case 'IDLE':
                return (
                    <div className="video-placeholder-content">
                        <span>Click 'Start' to begin translation</span>
                    </div>
                );
            case 'CONNECTING':
                return (
                    <div className="video-placeholder-content">
                        <div className="spinner"></div>
                        <span>Connecting...</span>
                    </div>
                );
            case 'ERROR':
                return (
                    <div className="video-placeholder-content error">
                        <span>Connection error. Please try again.</span>
                    </div>
                );
            case 'LISTENING':
                return <video ref={ref} autoPlay playsInline muted />;
            case 'PROCESSING':
                return (
                    <div className="video-placeholder-content">
                        <div className="spinner"></div>
                        <span>Elaborazione traduzione...</span>
                    </div>
                );
            default:
                return (
                    <div className="video-placeholder-content">
                        <span>Click 'Start' to begin translation</span>
                    </div>
                );
        }
    };

    return (
        <div className="video-container">
            {renderContent()}
        </div>
    );
});

export default VideoDisplay;