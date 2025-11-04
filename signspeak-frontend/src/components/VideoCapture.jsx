import React, { useEffect, forwardRef } from 'react'; // Import forwardRef

const VideoCapture = forwardRef(({ stream }, ref) => {

    useEffect(() => {
        if (ref.current && stream) {
            // Connect the webcam stream to the <video> tag to display it
            ref.current.srcObject = stream;
        }
    }, [stream, ref]);

    return (
        <div className="video-container">
            <video ref={ref} autoPlay playsInline muted />
        </div>
    );
});

export default VideoCapture;