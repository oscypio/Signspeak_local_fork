import React, { useState, useRef } from 'react';
import "../styles/TranslationPage.css";

import { getWebcamStream } from '../services/WebcamCaptureService';
import { MediaPipeService } from '../services/MediaPipeService';
import { WebSocketManager } from '../services/WebSocketManager';
import ControlPanel from '../components/ControlPanel';
import VideoDisplay from '../components/VideoDisplay';
import TextDisplay from '../components/TextDisplay';
import { HiMiniSpeakerWave } from "react-icons/hi2";
import { FaEdit } from "react-icons/fa";
import { MdOutlineTextIncrease, MdOutlineTextDecrease } from "react-icons/md";



function TranslationPage() {

    const [mediaStream, setMediaStream] = useState(null);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const videoRef = useRef(null);
    const mediaPipeService = useRef(new MediaPipeService()).current;
    const wsManager = useRef(new WebSocketManager()).current;
    const [status, setStatus] = useState('IDLE');
    const [translatedText, setTranslatedText] = useState('');

    /**
     * Handles incoming messages from the WebSocket.
     * Expects either status messages (plain text) or translations (JSON).
     */
    const handleBackendMessage = (message) => {
        if (message && typeof message === 'string' && message.trim().startsWith('{')) {

            try {
                const data = JSON.parse(message);

                if (data.text) {
                    setTranslatedText(data.text);

                    console.log("Final translation received. Disconnecting.");
                    wsManager.disconnect();

                    setStatus('IDLE');
                }
            } catch (error) {
                console.error("Failed to parse malformed JSON message:", error, message);
            }

        }
    };

    /**
     * Callback for MediaPipe. Sends landmarks to the WebSocketManager.
     */
    const handleLandmarks = (results, timestamp) => {
        if (results) {
            wsManager.sendHandData(results, timestamp);
        }
    };

    /**
     * Starts the entire process: connection, webcam, and MediaPipe.
     */
    const handleStart = async () => {
        console.log("Connecting to Websocket...");
        setStatus('CONNECTING');
        setTranslatedText('');

        wsManager.connect({
            onOpen: async () => {
                console.log("Websocket Connected. Requesting webcam...");
                const stream = await getWebcamStream();

                if (stream) {
                    setMediaStream(stream);
                    setIsWebcamOn(true);
                    setStatus('LISTENING');

                    console.log("Initializing MediaPipe...");
                    await mediaPipeService.initialize(handleLandmarks);

                    setTimeout(() => {
                        if (videoRef.current) {
                            mediaPipeService.startProcessing(videoRef.current);
                            console.log("MediaPipe processing started.");
                        }
                    }, 100);
                } else {
                    console.log("Webcam permission denied.");
                    setStatus('IDLE');
                    wsManager.disconnect();
                }
            },
            onMessage: handleBackendMessage,
            onError: (error) => {
                console.error("Failed to connect to WebSocket.", error);
                setStatus('ERROR');
            },
            onClose: () => {
                setStatus((currentState) => {
                    if (currentState !== 'IDLE' && currentState !== 'ERROR') {
                        console.log("WebSocket connection closed by server.");
                        return 'ERROR';
                    }
                    return currentState;
                });
            }
        });
    };

    /**
     * Stops the webcam and MediaPipe, then waits for the final translation.
     */
    const handleStop = () => {
        console.log("Stopping webcam... waiting for final translation");
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        setMediaStream(null);
        setIsWebcamOn(false);

        mediaPipeService.stopProcessing();

        setStatus('PROCESSING')
    };

    /**
     * Single handler for the Start/Stop button.
     */
    const toggleWebcam = () => {
        if (isWebcamOn) {
            handleStop();
        } else {
            handleStart();
        }
    };

    /**
     * Simulation function for testing the UI.
     */
    const simulateReceive = () => {
        const mockMessage = JSON.stringify({ text: "This is a simulated translation from the backend." });
        handleBackendMessage(mockMessage);
    };

    /**
     * add new states for buttons: increase/decrease text size, read aloud, edit text
     */
    const [fontSize, setFontSize] = useState(18);
    const [isEditable, setIsEditable] = useState(false);

    /**
     * edit size functions
     */
    const increaseFont = () => setFontSize((s) => s + 2);
    const decreaseFont = () => setFontSize((s) => Math.max(12, fontSize - 2));

    /**
     * Edit text function
     */
    const textEdit = () => setIsEditable((prev) => !prev);

    return (
        <main className="translation-content">

            <section className="video-section">
                <VideoDisplay
                    ref={videoRef}
                    status={status}
                    stream={mediaStream}
                />
                <ControlPanel
                    onClick={toggleWebcam}
                    isWebcamOn={isWebcamOn}
                    status={status}
                />
            </section>

            <section className="translation-section">
                <div className="translation-box">
                    <div
                        className="result-text"
                        contentEditable={isEditable}
                        suppressContentEditableWarning={true}
                        style={{ fontSize: `${fontSize}px` }}
                        onInput={(e) => setTranslatedText(e.target.textContent)}
                    >
                        {translatedText || ""}
                    </div>

                    {/* Botones inferiores */}
                    <div className="translation-actions">
                        <button className="translated-text"><HiMiniSpeakerWave /></button>
                        <button className="increase-font" onClick={increaseFont}><MdOutlineTextIncrease /></button>
                        <button className="decrease-font" onClick={decreaseFont}><MdOutlineTextDecrease /></button>
                        <button className="text-edit" onClick={textEdit}><FaEdit /></button>
                    </div>
                </div>
            </section>


            <button onClick={simulateReceive} style={{marginTop: '1rem'}}>
                Simulate Backend Message
            </button>
        </main>
    );
}

export default TranslationPage;