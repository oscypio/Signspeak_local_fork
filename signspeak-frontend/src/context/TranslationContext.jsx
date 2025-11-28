import React, { createContext, useContext, useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { getWebcamStream } from '../services/WebcamCaptureService';
import { MediaPipeService } from '../services/MediaPipeService';
import { WebSocketManager } from '../services/WebSocketManager';
import PiPContent from '../components/PiPContent';
import { FiX } from "react-icons/fi";

const TranslationContext = createContext();

export const useTranslation = () => useContext(TranslationContext);

const parseMeetingInput = (input) => {
    if (!input) return { id: null, type: null, url: null };
    const text = input.trim();
    // 1. GOOGLE MEET
    const meetRegex = /meet\.google\.com\/([a-z]{3}-[a-z]{4}-[a-z]{3})/;
    if (meetRegex.test(text)) return { id: text.match(meetRegex)[1], type: 'Google Meet', url: `https://meet.google.com/${text.match(meetRegex)[1]}` };
    if (/^[a-z]{3}-[a-z]{4}-[a-z]{3}$/.test(text)) return { id: text, type: 'Google Meet', url: `https://meet.google.com/${text}` };
    // 2. ZOOM
    const zoomRegex = /zoom\.us\/[jw]\/(\d+)/;
    if (zoomRegex.test(text)) return { id: "zoom-" + text.match(zoomRegex)[1], type: 'Zoom', url: text };
    if (/^\d{9,11}$/.test(text)) return { id: "zoom-" + text, type: 'Zoom', url: `https://zoom.us/j/${text}` };
    // 3. TEAMS
    if (text.includes('teams.microsoft.com/l/meetup-join/')) {
        try {
            const urlObj = new URL(text);
            const pathParts = urlObj.pathname.split('/');
            const joinIndex = pathParts.indexOf('meetup-join');
            if (joinIndex !== -1 && pathParts[joinIndex + 1]) {
                let rawId = decodeURIComponent(pathParts[joinIndex + 1]);
                const cleanId = "teams-" + rawId.replace(/[^a-zA-Z0-9]/g, '').substring(0, 30);
                return { id: cleanId, type: 'Microsoft Teams', url: text };
            }
        } catch (e) { console.error("Teams error", e); return { id: null, type: null, url: null }; }
    }

    return { id: null, type: null, url: null };
};

// Modal Component
const MeetConfigModal = ({ currentMeetingId, currentUserStatus, onSubmit, onClose, isPiPMode }) => {
    const [inputValue, setInputValue] = useState(currentMeetingId || '');
    const [tempUserStatus, setTempUserStatus] = useState(currentUserStatus);

    const parsedInfo = parseMeetingInput(inputValue);
    const isValid = !!parsedInfo.id;

    const handleSubmit = () => {
        if (!isValid) return;
        onSubmit(parsedInfo.id, tempUserStatus);
        onClose();
    };

    const containerWidth = isPiPMode ? '90%' : '420px';

    return (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.8)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 10000 }}>
            <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '12px', width: containerWidth, maxWidth: '420px', boxShadow: '0 10px 25px rgba(0,0,0,0.3)' }}>
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom: '15px'}}>
                    <h3 style={{margin: 0, color:'#183651', fontSize: '1.2rem'}}>Meeting Setup</h3>
                    <button onClick={onClose} style={{background:'none', border:'none', fontSize:'20px', cursor:'pointer', color:'#999'}}><FiX/></button>
                </div>

                <div style={{ marginBottom: '15px' }}>
                    <label style={{display:'block', marginBottom:'5px', fontWeight:'600', color:'#333', fontSize:'0.9rem'}}>Link or ID:</label>
                    <div style={{position: 'relative', display: 'flex', alignItems: 'center'}}>
                        <input
                            type="text"
                            placeholder="Paste link..."
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            style={{width:'100%', padding:'10px 35px 10px 10px', border: `2px solid ${isValid ? '#10b981' : '#ddd'}`, borderRadius:'6px', fontSize: '14px', boxSizing: 'border-box', outline: 'none'}}
                        />
                        <div style={{position: 'absolute', right: '10px', pointerEvents: 'none'}}>
                            {isValid ? <span style={{color: '#10b981'}}>✓</span> : (inputValue && <span style={{color: '#ef4444'}}>✕</span>)}
                        </div>
                    </div>
                    {isValid && <div style={{fontSize:'11px', color:'#10b981', marginTop:'4px'}}>Detected: {parsedInfo.type}</div>}
                </div>

                <div style={{ marginBottom: '20px' }}>
                    <label style={{display:'block', marginBottom:'5px', fontWeight:'600', color:'#333', fontSize:'0.9rem'}}>Role:</label>
                    <select value={tempUserStatus} onChange={(e) => setTempUserStatus(e.target.value)} style={{width:'100%', padding:'10px', border:'1px solid #ddd', borderRadius:'6px', fontSize: '14px'}}>
                        <option value="DEAF">Signer (Deaf)</option>
                        <option value="NORMAL">Listener (Hearing)</option>
                    </select>
                </div>

                <button onClick={handleSubmit} disabled={!isValid} style={{width:'100%', padding:'10px', background: isValid ? '#183651' : '#ccc', color:'white', border:'none', borderRadius:'8px', fontWeight:'bold', cursor: isValid ? 'pointer' : 'not-allowed'}}>
                    {isPiPMode ? "Update" : "Launch PiP"}
                </button>
            </div>
        </div>
    );
};

export const TranslationProvider = ({ children }) => {
    const copyStyles = (sourceDoc, targetDoc) => {
        Array.from(sourceDoc.styleSheets).forEach((styleSheet) => {
            try {
                if (styleSheet.cssRules) {
                    const newStyleEl = targetDoc.createElement('style');
                    Array.from(styleSheet.cssRules).forEach((cssRule) => {
                        newStyleEl.appendChild(targetDoc.createTextNode(cssRule.cssText));
                    });
                    targetDoc.head.appendChild(newStyleEl);
                } else if (styleSheet.href) {
                    const newLinkEl = targetDoc.createElement('link');
                    newLinkEl.rel = 'stylesheet';
                    newLinkEl.href = styleSheet.href;
                    targetDoc.head.appendChild(newLinkEl);
                }
            } catch (e) {
                console.error("Skipped copying a stylesheet due to CORS security rules: ", e);
            }
        });
    };

    const [meetingId, setMeetingId] = useState(() => localStorage.getItem('signSpeak_meetingId') || '');
    const [userStatus, setUserStatus] = useState(() => localStorage.getItem('signSpeak_userStatus') || 'DEAF');

    const [mediaStream, setMediaStream] = useState(null);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const [status, setStatus] = useState('IDLE');
    const [translatedText, setTranslatedText] = useState('');
    const [partialWords, setPartialWords] = useState([]);
    const [pipWindow, setPipWindow] = useState(null);

    const processingVideoRef = useRef(null);
    const videoRef = useRef(null);
    const mediaPipeService = useRef(new MediaPipeService()).current;
    const wsManager = useRef(new WebSocketManager()).current;
    const pipWindowRef = useRef(null);

    const translatedTextRef = useRef(translatedText);
    const meetingIdRef = useRef(meetingId);

    const [showConfigModal, setShowConfigModal] = useState(false);

    useEffect(() => {
        meetingIdRef.current = meetingId;
        if (meetingId)
            localStorage.setItem('signSpeak_meetingId', meetingId);
        else
            localStorage.removeItem('signSpeak_meetingId');
    }, [meetingId]);

    useEffect(() => {
        localStorage.setItem('signSpeak_userStatus', userStatus);
    }, [userStatus]);

    useEffect(() => { translatedTextRef.current = translatedText; }, [translatedText]);

    const handleBackendMessage = (message) => {
        if (message && typeof message === 'string' && message.trim().startsWith('{')) {
            try {
                const data = JSON.parse(message);
                if (data.meetingId && meetingIdRef.current && data.meetingId !== meetingIdRef.current) return;

                if (data.type === "AUDIO_COMMAND" && data.text) {
                    triggerLocalSpeech(data.text);
                } else if (data.type === "PARTIAL" && data.words) {
                    setPartialWords(data.words);
                } else if (data.type === "FINAL" && data.text) {
                    setTranslatedText(data.text);
                    setPartialWords([]);
                } else if (data.text && !data.type) {
                    setTranslatedText(data.text);
                }
            } catch (error) { console.error("JSON Error:", error); }
        }
    };

    const handleLandmarks = (results, timestamp) => {
        if (results) wsManager.sendHandData(results, timestamp, { meetingId: meetingIdRef.current, userStatus });
    };

    const connectToWebSocket = async (overrideMeetingId = null, overrideUserStatus = null, autoStartCamera = true) => {
        const currentId = overrideMeetingId ?? meetingIdRef.current ?? meetingId ?? "";
        let currentStatus = overrideUserStatus || userStatus;

        if (!currentId) {
            console.log("No Meeting ID detected (Local Mode): Force Webcam On (DEAF Mode)");
            currentStatus = 'DEAF';
            autoStartCamera = true;
        }

        console.log(`[Connection Start] ID: ${currentId}, Role: ${currentStatus}, , AutoStartCam: ${autoStartCamera}`);


        if (autoStartCamera) {
            setStatus('CONNECTING');
        }

        setTranslatedText('');
        setPartialWords([]);

        wsManager.connect({
            onOpen: async () => {
                if (currentStatus === 'DEAF') {
                    if (autoStartCamera) {
                        const stream = await getWebcamStream();
                        if (stream) {
                            setMediaStream(stream);
                            setIsWebcamOn(true);
                            setStatus('LISTENING');
                            await mediaPipeService.initialize(handleLandmarks);

                            if (processingVideoRef.current) {
                                processingVideoRef.current.srcObject = stream;

                                processingVideoRef.current.onloadedmetadata = () => {
                                    processingVideoRef.current.play();
                                    mediaPipeService.startProcessing(processingVideoRef.current);
                                };
                            }
                        } else {
                            setStatus('IDLE');
                            wsManager.disconnect();
                        }
                    } else {
                        console.log("PiP Deaf Mode: Connesso al BE. In attesa di Start manuale.");
                        setIsWebcamOn(false);
                        setStatus('IDLE');
                    }
                } else {
                    console.log("Connecting as Listener (No Camera)");
                    setIsWebcamOn(false);
                    setStatus('LISTENING');
                }
            },
            onMessage: handleBackendMessage,
            onError: () => setStatus('ERROR'),
            onClose: () => setStatus('IDLE')
        }, currentId);
    };

    const handleStart = async () => {
        connectToWebSocket();
    };

    const handleConfigSubmit = (newId, newStatus) => {
        setMeetingId(newId);
        setUserStatus(newStatus);

        if (!pipWindowRef.current) {
            enterPiP(newStatus);
        }

        if (isWebcamOn) handleStop();

        setShowConfigModal(false);

        setTimeout(() => {
            const shouldAutoStart = false;
            connectToWebSocket(newId, newStatus, shouldAutoStart);
        }, 100);
    };

    const handleStop = () => {
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        setMediaStream(null);
        setIsWebcamOn(false);
        mediaPipeService.stopProcessing();
        wsManager.disconnect();
        setPartialWords([]);
        setStatus('IDLE');
    };

    const toggleWebcam = () => isWebcamOn ? handleStop() : handleStart();

    const triggerLocalSpeech = (text) => {
        const synth = window.speechSynthesis;
        if (!synth) { console.error("No TTS support"); return; }

        if (synth.speaking) synth.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';

        const playWithVoice = () => {
            const voices = synth.getVoices();
            const voice = voices.find(v => v.name.includes('Google US English')) || voices.find(v => v.lang.startsWith('en')) || voices[0];
            if (voice) utterance.voice = voice;
            synth.speak(utterance);
        };

        if (synth.getVoices().length === 0) synth.onvoiceschanged = () => { synth.onvoiceschanged = null; playWithVoice(); };
        else playWithVoice();
    };

    const readAloud = () => {
        const text = translatedTextRef.current || "No text";
        if (meetingIdRef.current && wsManager.stompClient && wsManager.stompClient.connected)  {
            wsManager.sendAudioTrigger(text, meetingIdRef.current);
        }  else {
            triggerLocalSpeech(text);
        }
    };

    // PiP logic
    const enterPiP = async (overrideStatus = null) => {
        if (!('documentPictureInPicture' in window)) return alert("No PiP support");

        const currentStatus = overrideStatus || userStatus;

        try {
            const pipHeight = currentStatus === 'DEAF' ? 700 : 450;
            const pipWidth = 400;

            // @ts-ignore
            const newPipWindow = await window.documentPictureInPicture.requestWindow({
                width: pipWidth,
                height: pipHeight
            });

            //newPipWindow.resizeTo(pipWidth, pipHeight);

            copyStyles(document, newPipWindow.document);

            const pipStyle = newPipWindow.document.createElement('style');
            pipStyle.textContent = `
                .pip-wrapper .start-btn {
                    width: auto !important; 
                    min-width: unset !important;
                    padding: 8px 40px !important; /* Padding ridotto */
                    margin: 0 auto 20px auto !important; 
                    display: block; 
                    font-size: 1rem !important;
                }
                .pip-wrapper .video-section {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
            `;
            newPipWindow.document.head.appendChild(pipStyle);

            setPipWindow(newPipWindow);
            pipWindowRef.current = newPipWindow;

            newPipWindow.addEventListener('pagehide', () => {
                setPipWindow(null);
                pipWindowRef.current = null;
                handleStop();
                setMeetingId('');
            });

        } catch (err) { console.error(err); }
    };

    const [fontSize, setFontSize] = useState(18);
    const increaseFont = () => setFontSize(s => s + 2);
    const decreaseFont = () => setFontSize(s => Math.max(12, s - 2));

    const value = {
        meetingId, setMeetingId,
        userStatus, setUserStatus,
        mediaStream, isWebcamOn, status,
        translatedText, partialWords,
        handleStart, handleStop, toggleWebcam,
        readAloud, enterPiP,
        videoRef,
        fontSize, increaseFont, decreaseFont,
        pipWindow,
        showConfigModal, setShowConfigModal
    };

    return (
        <TranslationContext.Provider value={value}>
            {children}
            <video
                ref={processingVideoRef}
                style={{
                    position: 'fixed',
                    top: '-1000px',
                    left: '-1000px',
                    width: '640px',
                    height: '480px',
                    opacity: 0,
                    pointerEvents: 'none'
                }}
                autoPlay
                muted
                playsInline
            />

            {showConfigModal && (
                pipWindow ?
                    createPortal(
                        <MeetConfigModal
                            currentMeetingId={meetingId}
                            currentUserStatus={userStatus}
                            onSubmit={handleConfigSubmit}
                            onClose={() => setShowConfigModal(false)}
                            isPiPMode={true}
                        />, pipWindow.document.body
                    ) : (
                        <MeetConfigModal
                            currentMeetingId={meetingId}
                            currentUserStatus={userStatus}
                            onSubmit={handleConfigSubmit}
                            onClose={() => setShowConfigModal(false)}
                        />
                    )
            )}

            {pipWindow && createPortal(
                <PiPContent
                    isDeaf={userStatus === 'DEAF'}
                    stream={mediaStream}
                    isWebcamOn={isWebcamOn}
                    toggleWebcam={toggleWebcam}
                    translatedText={translatedText}
                    partialWords={partialWords}
                    fontSize={fontSize}
                    increaseFont={increaseFont}
                    decreaseFont={decreaseFont}
                    readAloud={readAloud}
                />,
                pipWindow.document.body
            )}
        </TranslationContext.Provider>
    );
};