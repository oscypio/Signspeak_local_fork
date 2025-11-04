import React, { useState, useRef } from 'react';

import { getWebcamStream } from '../services/WebcamCaptureService';
import { MediaPipeService } from '../services/MediaPipeService';
import { WebSocketManager } from '../services/WebSocketManager';
import ControlPanel from '../components/ControlPanel';
import VideoCapture from '../components/VideoCapture';

const BACKEND_URL = "ws://localhost:8080/ws/landmarks";

function TranslationPage() {

    const [mediaStream, setMediaStream] = useState(null);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const videoRef = useRef(null);
    const mediaPipeService = useRef(new MediaPipeService()).current;
    const wsManager = useRef(new WebSocketManager()).current;

    const handleBackendMessage = (message) => {
        console.log("Message from backend:", message);
    };

    const handleLandmarks = (results, timestamp) => {
        if (results) {
            wsManager.sendHandData(results, timestamp);
        }
    };

    const handleStart = async () => {
        console.log("Starting webcam...");
        const stream = await getWebcamStream();
        if (stream) {
            setMediaStream(stream);
            setIsWebcamOn(true);
            console.log("Connecting to WebSocket backend...");
            wsManager.connect(BACKEND_URL, handleBackendMessage);
            console.log("Initializing MediaPipe...");
            await mediaPipeService.initialize(handleLandmarks);
            setTimeout(() => {
                if (videoRef.current) {
                    mediaPipeService.startProcessing(videoRef.current);
                    console.log("MediaPipe processing started.");
                }
            }, 500);
        }
    };

    const handleStop = () => {
        console.log("Stopping webcam...");
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        setMediaStream(null);
        setIsWebcamOn(false);
        mediaPipeService.stopProcessing();
        console.log("Disconnecting from WebSocket...");
        wsManager.disconnect();
    };

    const toggleWebcam = () => {
        if (isWebcamOn) {
            handleStop();
        } else {
            handleStart();
        }
    };

    return (
        <main className="main-content">
            <section className="video-section">
                <VideoCapture ref={videoRef} stream={mediaStream} />
                <ControlPanel
                    onClick={toggleWebcam}
                    isWebcamOn={isWebcamOn}
                />
            </section>

        </main>
    );
}

export default TranslationPage;
