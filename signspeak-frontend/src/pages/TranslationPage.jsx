import React, { useState, useRef, useEffect } from 'react';
import "../styles/TranslationPage.css";

import { getWebcamStream } from '../services/WebcamCaptureService';
import { MediaPipeService } from '../services/MediaPipeService';
import { WebSocketManager } from '../services/WebSocketManager';
import ControlPanel from '../components/ControlPanel'; // Nota: ControlPanel contiene il tasto Start
import VideoDisplay from '../components/VideoDisplay';

// Icone
import { HiMiniSpeakerWave } from "react-icons/hi2";
import { FaEdit } from "react-icons/fa";
import { MdOutlineTextIncrease, MdOutlineTextDecrease } from "react-icons/md";
import { FiSettings, FiX } from "react-icons/fi";

// =========================================================================
// === VALIDAZIONE GOOGLE MEET ===
// =========================================================================

const validateAndExtractMeetingId = (input) => {
    if (!input) return null;
    const cleanInput = input.trim().toLowerCase();

    // Pattern esatto: 3 lettere - 4 lettere - 3 lettere (es. pzo-fddt-emi)
    const idPattern = /[a-z]{3}-[a-z]{4}-[a-z]{3}/;

    // Caso 1: Link intero
    if (cleanInput.includes('meet.google.com/')) {
        const match = cleanInput.match(/meet\.google\.com\/([a-z]{3}-[a-z]{4}-[a-z]{3})/);
        return match ? match[1] : null;
    }

    // Caso 2: Solo ID
    if (cleanInput.length === 12 && idPattern.test(cleanInput)) {
        return cleanInput;
    }

    return null;
};


// =========================================================================
// === STILI E COMPONENTI PER IL MODALE E PIP ===
// =========================================================================

const modalOverlayStyle = { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.6)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 100 };
const modalContentStyle = { backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)', width: '400px' };
const inputStyle = { padding: '10px', width: '100%', border: '1px solid #ddd', borderRadius: '4px', boxSizing: 'border-box' };
const labelStyle = { fontSize: '14px', display: 'block', marginBottom: '5px', color: '#333' };
const submitButtonStyle = { padding: '10px 15px', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', width: '100%' };
const closeButtonStyle = { background: 'none', border: 'none', fontSize: '18px', cursor: 'pointer', color: '#aaa' };

// Componente Modale
const GoogleMeetConfigModal = ({ currentMeetingId, currentUserStatus, onSubmit, onClose }) => {
    const [tempMeetingId, setTempMeetingId] = useState(currentMeetingId);
    const [tempUserStatus, setTempUserStatus] = useState(currentUserStatus);

    const handleSubmit = () => {
        const validId = validateAndExtractMeetingId(tempMeetingId);
        if (!validId) {
            alert("Per favore inserisci un link Google Meet valido (https://meet.google.com/...) oppure un ID nel formato 'xxx-xxxx-xxx'.");
            return;
        }
        onSubmit(validId, tempUserStatus);
        onClose();
    };

    return (
        <div style={modalOverlayStyle}>
            <div style={modalContentStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ color: '#333' }}>Google Meet Configuration</h3>
                    <button onClick={onClose} style={closeButtonStyle}><FiX /></button>
                </div>
                <div style={{ marginBottom: '15px' }}>
                    <label style={labelStyle}>Meeting Link/ID (Obbligatorio per PiP):</label>
                    <input
                        type="text"
                        placeholder="Es. https://meet.google.com/abc-defg-hij"
                        value={tempMeetingId}
                        onChange={(e) => setTempMeetingId(e.target.value)}
                        style={inputStyle}
                    />
                    <p style={{fontSize: '11px', color: '#888', marginTop: '4px'}}>Accetta link completi o solo l'ID (es. abc-defg-hij)</p>
                </div>
                <div style={{ marginBottom: '20px' }}>
                    <label style={labelStyle}>My Status:</label>
                    <select value={tempUserStatus} onChange={(e) => setTempUserStatus(e.target.value)} style={inputStyle}>
                        <option value="DEAF">Deaf User (Signing)</option>
                        <option value="NORMAL">Hearing Participant (Reading)</option>
                    </select>
                </div>
                <button onClick={handleSubmit} style={submitButtonStyle}>Open PiP Mode</button>
            </div>
        </div>
    );
};

// CSS PiP (Stile Figma)
const getPipStyles = () => `
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    body { margin: 0; padding: 0; font-family: 'Inter', sans-serif; background-color: #ffffff; color: #0d2538; display: flex; flex-direction: column; height: 100vh; }
    
    .pip-header { 
        background-color: #E9F6FA; 
        padding: 10px 20px; 
        display: flex; align-items: center; 
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05); 
        height: 60px; box-sizing: border-box; 
    }
    .pip-header-logo { height: 40px; width: auto; }

    .pip-container { padding: 20px; display: flex; flex-direction: column; flex-grow: 1; box-sizing: border-box; align-items: center; overflow: hidden; }

    video { width: 100%; height: 200px; background: #000; border-radius: 6px; margin-bottom: 20px; object-fit: cover; transform: scaleX(-1); }

    .pip-control-button { 
        background-color: #183651; color: #fff; 
        padding: 10px 40px; font-size: 1.1rem; 
        border-radius: 15px; border: none; 
        cursor: pointer; font-weight: 500; margin-bottom: 20px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: opacity 0.2s; 
    }
    .pip-control-button:hover { opacity: 0.9; }
    .pip-control-button:disabled { background-color: #ccc; cursor: not-allowed; }

    .pip-translation-card { background-color: #c7e8ef; padding: 15px; border-radius: 14px; width: 100%; box-sizing: border-box; display: flex; flex-direction: column; flex-grow: 1; }

    #pip-translation { background-color: #e8f8fb; border-radius: 14px; padding: 15px; font-size: 1.1rem; color: #0d2538; flex-grow: 1; overflow-y: auto; margin-bottom: 10px; min-height: 80px; }

    .pip-translation-actions { display: flex; justify-content: space-around; align-items: center; padding-top: 5px; }
    
    .action-button { width: 45px; height: 45px; display: flex; align-items: center; justify-content: center; border: none; background: transparent; cursor: pointer; color: #0d2538; transition: transform 0.1s; }
    .action-button svg { width: 28px; height: 28px; }
    .action-button:hover { transform: scale(1.1); }

    .pip-status-bar { color: #666; margin-bottom: 10px; font-weight: 500; text-align: center; }
`;


// =========================================================================
// === COMPONENTE PRINCIPALE ===
// =========================================================================

function TranslationPage() {

    const [mediaStream, setMediaStream] = useState(null);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const videoRef = useRef(null);
    const mediaPipeService = useRef(new MediaPipeService()).current;
    const wsManager = useRef(new WebSocketManager()).current;
    const [status, setStatus] = useState('IDLE');
    const [translatedText, setTranslatedText] = useState('');

    const [fontSize, setFontSize] = useState(18);

    const [showModal, setShowModal] = useState(false);
    const [meetingId, setMeetingId] = useState('');
    const [userStatus, setUserStatus] = useState('DEAF');
    const [isPiP, setIsPiP] = useState(false);
    const pipWindowRef = useRef(null);

    // ---------------------------------------------
    // LOGICA ORIGINALE
    // ---------------------------------------------
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

    const handleLandmarks = (results, timestamp) => {
        if (results) {
            const metaData = meetingId ? { meetingId, userStatus } : {};
            wsManager.sendHandData(results, timestamp, metaData);
        }
    };

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
                    if (currentState !== 'IDLE' && currentState !== 'ERROR') return 'ERROR';
                    return currentState;
                });
            }
        }, meetingId);
    };

    const handleStop = () => {
        console.log("Stopping webcam...");
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        setMediaStream(null);
        setIsWebcamOn(false);
        mediaPipeService.stopProcessing();
        setStatus('PROCESSING');
    };

    const toggleWebcam = () => {
        if (isWebcamOn) handleStop();
        else handleStart();
    };

    const simulateReceive = () => {
        const mockMessage = JSON.stringify({ text: "This is a simulated translation from the backend." });
        handleBackendMessage(mockMessage);
    };

    const increaseFont = () => setFontSize((s) => s + 2);
    const decreaseFont = () => setFontSize((s) => Math.max(12, s - 2));

    const readAloud = () => {
        const textToRead = translatedText || "No text to read";
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(textToRead);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        } else {
            alert("Text-to-speech not supported in this browser.");
        }
    };


    // ---------------------------------------------
    // NUOVA LOGICA PIP
    // ---------------------------------------------

    // 1. Enter PiP
    const enterPiP = async (targetMeetingId, targetUserStatus) => {
        if (!('documentPictureInPicture' in window)) {
            alert("Il tuo browser non supporta il PiP API.");
            return;
        }

        if (pipWindowRef.current) pipWindowRef.current.close();

        const isDeaf = targetUserStatus === 'DEAF';

        try {
            // @ts-ignore
            const newPipWindow = await window.documentPictureInPicture.requestWindow({
                width: 400,
                height: isDeaf ? 700 : 450
            });

            const style = newPipWindow.document.createElement('style');
            style.textContent = getPipStyles();
            newPipWindow.document.head.appendChild(style);

            // HEADER
            const header = newPipWindow.document.createElement('div');
            header.className = 'pip-header';
            header.innerHTML = `<img src="/Logo.png" alt="Logo" class="pip-header-logo" />`;
            newPipWindow.document.body.appendChild(header);

            // CONTAINER
            const container = newPipWindow.document.createElement('div');
            container.className = 'pip-container';

            // ICONE SVG
            const iconSpeaker = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0 001.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276 2.561-1.06V4.06zM18.584 9a.75.75 0 01.724 1.06 3.5 3.5 0 000 3.88.75.75 0 11-1.372.612 5 5 0 010-5.164A.75.75 0 0118.584 9z" /></svg>`;
            const iconPlus = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M5.625 18.75h12.75V17.5h-12.75v1.25Zm6.65-4.575L14.75 9.7l2.475 4.475h1.45l-3.2-5.75h-1.45l-3.2 5.75h1.45Zm-7.9 0L6.85 5.5l2.475 8.675h1.45l-3.2-11.2h-1.45L2.925 14.175h1.45Zm5.975-6.525L11.5 10.95h2.1L12.55 7.65h-2.2ZM6.125 8.925l1.075 3.85h-2.15l1.075-3.85Z" /></svg>`;
            const iconMinus = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M5.625 18.75h12.75V17.5h-12.75v1.25Zm6.65-4.575L14.75 9.7l2.475 4.475h1.45l-3.2-5.75h-1.45l-3.2 5.75h1.45Zm-7.9 0L6.85 5.5l2.475 8.675h1.45l-3.2-11.2h-1.45L2.925 14.175h1.45Zm5.975-6.525L11.5 10.95h2.1L12.55 7.65h-2.2ZM6.125 8.925l1.075 3.85h-2.15l1.075-3.85Z" /></svg>`;

            let htmlContent = '';

            if (isDeaf) {
                htmlContent += `
                    <video autoplay playsinline muted></video>
                    <button class="pip-control-button" id="pip-btn-toggle">Start</button>
                    <div class="pip-translation-card">
                        <div id="pip-translation" style="font-size: ${fontSize}px">Press Start...</div>
                        <div class="pip-translation-actions">
                            <button class="action-button" id="pip-action-speak">${iconSpeaker}</button>
                            <button class="action-button" id="pip-action-plus">${iconPlus}</button>
                            <button class="action-button" id="pip-action-minus">${iconMinus}</button>
                        </div>
                    </div>
                `;
            } else {
                htmlContent += `
                    <div class="pip-status-bar">Listening Mode Active</div>
                    <div class="pip-translation-card">
                        <div id="pip-translation" style="font-size: ${fontSize}px">Waiting for translation...</div>
                        <div class="pip-translation-actions">
                            <button class="action-button" id="pip-action-speak">${iconSpeaker}</button>
                            <button class="action-button" id="pip-action-plus">${iconPlus}</button>
                            <button class="action-button" id="pip-action-minus">${iconMinus}</button>
                        </div>
                    </div>
                `;
            }

            container.innerHTML = htmlContent;
            newPipWindow.document.body.appendChild(container);

            if (isDeaf) {
                const btn = newPipWindow.document.getElementById('pip-btn-toggle');
                if (btn) btn.onclick = () => toggleWebcam();
            }

            newPipWindow.document.getElementById('pip-action-speak')?.addEventListener('click', readAloud);
            newPipWindow.document.getElementById('pip-action-plus')?.addEventListener('click', increaseFont);
            newPipWindow.document.getElementById('pip-action-minus')?.addEventListener('click', decreaseFont);

            newPipWindow.addEventListener('pagehide', () => {
                setIsPiP(false);
                pipWindowRef.current = null;
                if (isWebcamOn) handleStop();
            });

            pipWindowRef.current = newPipWindow;
            setIsPiP(true);

        } catch (err) {
            console.error(err);
        }
    };

    // 2. Submit Config
    const handleConfigSubmit = (newMeetingId, newUserStatus) => {
        setMeetingId(newMeetingId);
        setUserStatus(newUserStatus);
        if (isWebcamOn) handleStop();
        enterPiP(newMeetingId, newUserStatus);
    };


    // ---------------------------------------------
    // SYNC EFFECTS
    // ---------------------------------------------

    useEffect(() => {
        if (pipWindowRef.current && userStatus === 'DEAF' && mediaStream) {
            const pipVideo = pipWindowRef.current.document.querySelector('video');
            if (pipVideo && pipVideo.srcObject !== mediaStream) {
                pipVideo.srcObject = mediaStream;
                pipVideo.play().catch(console.log("Auto-play prevented"));
            }
        }
    }, [mediaStream, isPiP, userStatus]);

    useEffect(() => {
        if (pipWindowRef.current) {
            const pipText = pipWindowRef.current.document.getElementById('pip-translation');
            if (pipText) {
                pipText.textContent = translatedText || (userStatus === 'DEAF' ? "Listening..." : "Waiting...");
                pipText.style.fontSize = `${fontSize}px`;
            }
        }
    }, [translatedText, fontSize, isPiP]);

    useEffect(() => {
        if (pipWindowRef.current && userStatus === 'DEAF') {
            const btn = pipWindowRef.current.document.getElementById('pip-btn-toggle');
            if (btn) {
                if (status === 'CONNECTING') {
                    btn.textContent = 'Connecting...';
                    btn.disabled = true;
                } else if (status === 'PROCESSING') {
                    btn.textContent = 'Processing...';
                    btn.disabled = true;
                } else {
                    btn.textContent = isWebcamOn ? 'Stop' : 'Start';
                    btn.disabled = false;
                }
                btn.onclick = () => toggleWebcam();
            }
        }
    }, [status, isWebcamOn, isPiP]);


    // ---------------------------------------------
    // RENDER PRINCIPALE
    // ---------------------------------------------
    return (
        <main className="translation-content">

            {/* MODALE DI CONFIGURAZIONE (Invisibile fino all'attivazione) */}
            {showModal && (
                <GoogleMeetConfigModal
                    currentMeetingId={meetingId}
                    currentUserStatus={userStatus}
                    onSubmit={handleConfigSubmit}
                    onClose={() => setShowModal(false)}
                />
            )}

            {/* SEZIONE VIDEO (SINISTRA) */}
            <section className="video-section">
                <VideoDisplay
                    ref={videoRef}
                    status={status}
                    stream={mediaStream}
                />

                {/* --- BOTTONI START E PIP SETUP ALLINEATI --- */}
                <div className="buttons-container">
                    <ControlPanel
                        onClick={toggleWebcam}
                        isWebcamOn={isWebcamOn}
                        status={status}
                    />

                    <button
                        className="setup-pip-btn"
                        onClick={() => setShowModal(true)}
                    >
                        <FiSettings style={{ marginRight: '8px' }} /> {isPiP ? 'PiP Active' : 'Setup PiP'}
                    </button>
                </div>
            </section>

            {/* SEZIONE TRADUZIONE (DESTRA) */}
            <section className="translation-section">
                <div className="translation-box">
                    <div
                        className="result-text"
                        suppressContentEditableWarning={true}
                        style={{ fontSize: `${fontSize}px` }}
                        onInput={(e) => setTranslatedText(e.target.textContent)}
                    >
                        {translatedText || ""}
                    </div>

                    <div className="translation-actions">
                        <button className="translated-text" onClick={readAloud}><HiMiniSpeakerWave /></button>
                        <button className="increase-font" onClick={increaseFont}><MdOutlineTextIncrease /></button>
                        <button className="decrease-font" onClick={decreaseFont}><MdOutlineTextDecrease /></button>
                    </div>
                </div>
            </section>

            <button onClick={simulateReceive} style={{ marginTop: '1rem', position: 'fixed', bottom: '10px', right: '10px', opacity: 0.5 }}>
                Simulate Backend Message
            </button>
        </main>
    );
}

export default TranslationPage;